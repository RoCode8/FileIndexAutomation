import base64
import re
import io
import openai
import html
import time
import pathlib
import tiktoken
from PyPDF2 import PdfReader, PdfWriter
from config import *

from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.identity import AzureDeveloperCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)


from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


MAX_SECTION_LENGTH = 1000
SENTENCE_SEARCH_LIMIT = 100
SECTION_OVERLAP = 100

open_ai_token_cache = {}
CACHE_KEY_TOKEN_CRED = 'openai_token_cred'
CACHE_KEY_CREATED_TIME = 'created_time'
CACHE_KEY_TOKEN_TYPE = 'token_type'

#Embedding batch support section
SUPPORTED_BATCH_AOAI_MODEL = {
    'text-embedding-ada-002': {
        'token_limit' : 8100,
        'max_batch_size' : 16
    }
}


def calculate_tokens_emb_aoai(input: str):
    encoding = tiktoken.encoding_for_model(openaimodelname)
    return len(encoding.encode(input))

def blob_name_from_file_page(filename, page = 0):

    file_name = pathlib.Path(filename).stem
    extension = "".join(pathlib.Path(filename).suffixes).lower()
    if extension == ".pdf":
        return f"{file_name}-{page}.pdf"
    else: 
        return filename
def upload_blobs(filename):

    # Get the blob client for the specific file
    blob_client = container_client.get_blob_client(filename)

    # Download the blob content
    blob_data = blob_client.download_blob().readall()
    stream = io.BytesIO(blob_data)

    # If file is PDF, split into pages and upload each page as a separate blob
    if pathlib.Path(filename).suffix.lower() == ".pdf":
        reader = PdfReader(stream)
        pages = reader.pages
        for i, page in enumerate(pages):
            page_blob_name = blob_name_from_file_page(filename, i)
            print(f"\tUploading blob for page {i} -> {page_blob_name}")
            
            page_stream = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(page_stream)
            page_stream.seek(0)

            split_container_client.upload_blob(page_blob_name, page_stream, overwrite=True)
    else:
        # For non-PDF files, upload directly
        split_container_client.upload_blob(blob_name_from_file_page(filename), stream, overwrite=True)

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def get_document_text(filename):
    offset = 0
    page_map = []
    formrecognizerservice = AZURE_FORMRECOGNIZER_SERVICE_NAME
    formrecognizer_creds = AzureKeyCredential(AZURE_FORMRECOGNIZER_ACCESS_KEY)

    # Initailizing blob client
    blob_client = container_client.get_blob_client(filename)

    # Read the blob content into memory
    blob_data = blob_client.download_blob().readall()
    stream = io.BytesIO(blob_data)
    
    print(f"Extracting text from '{filename}' using Azure Form Recognizer")

    form_recognizer_client = DocumentAnalysisClient(
        endpoint = f"https://{formrecognizerservice}.cognitiveservices.azure.com/", 
        credential = formrecognizer_creds, 
        headers = {"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
    
    poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document=stream)
    form_recognizer_results = poller.result()

    for page_num, page in enumerate(form_recognizer_results.pages):
        tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

        # mark all positions of the table spans in the page
        page_offset = page.spans[0].offset
        page_length = page.spans[0].length
        table_chars = [-1]*page_length
        for table_id, table in enumerate(tables_on_page):
            for span in table.spans:
                # replace all table spans with "table_id" in table_chars array
                for i in range(span.length):
                    idx = span.offset - page_offset + i
                    if idx >=0 and idx < page_length:
                        table_chars[idx] = table_id

        # build page text by replacing characters in table spans with table html
        page_text = ""
        added_tables = set()
        for idx, table_id in enumerate(table_chars):
            if table_id == -1:
                page_text += form_recognizer_results.content[page_offset + idx]
            elif table_id not in added_tables:
                page_text += table_to_html(tables_on_page[table_id])
                added_tables.add(table_id)

        page_text += " "
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)
        stream.seek(0) # resetting the stream position after use

    return page_map

def split_text(page_map, filename):
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]
    print(f"Splitting '{filename}' into sections")

    def find_page(offset):
        num_pages = len(page_map)
        for i in range(num_pages - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return num_pages - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
            print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP


    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))

def filename_to_id(filename):
    filename_ascii = re.sub("[^0-9a-zA-Z_-]", "_", filename)
    filename_hash = base64.b16encode(filename.encode('utf-8')).decode('ascii')
    return f"file-{filename_ascii}-{filename_hash}"

def create_sections(filename, page_map, use_vectors, embedding_deployment: str = None):
    file_id = filename_to_id(filename)
    for i, (content, pagenum) in enumerate(split_text(page_map, filename)):
        section = {
            "id": f"{file_id}-page-{i}",
            "content": content,
            # "category": category,
            "sourcepage": blob_name_from_file_page(filename, pagenum),
            "sourcefile": filename
        }
        if use_vectors:
            section["embedding"] = compute_embedding(content, embedding_deployment)
        yield section

def before_retry_sleep(retry_state):
    print("Rate limited on the OpenAI embeddings API, sleeping before retrying...")

@retry(retry=retry_if_exception_type(openai.error.RateLimitError), wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
def compute_embedding(text, embedding_deployment):
    refresh_openai_token()
    return openai.Embedding.create(engine=embedding_deployment, input=text)["data"][0]["embedding"]

@retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(15), before_sleep=before_retry_sleep)
def compute_embedding_in_batch(texts):
    refresh_openai_token()
    emb_response = openai.Embedding.create(engine=openaideployment, input=texts)
    return [data.embedding for data in emb_response.data]

def create_search_index(index):
    print(f"Ensuring search index {index} exists")
    index_client = SearchIndexClient(endpoint=f"https://{searchservice}.search.windows.net/",
                                     credential=search_creds)
    if index not in index_client.list_index_names():
        index = SearchIndex(
            name=index,
            fields=[
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String", analyzer_name="en.microsoft"),
                SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            hidden=False, searchable=True, filterable=False, sortable=False, facetable=False,
                            vector_search_dimensions=1536, vector_search_configuration="default"),
                SimpleField(name="category", type="Edm.String", filterable=True, facetable=True),
                SimpleField(name="sourcepage", type="Edm.String", filterable=True, facetable=True),
                SimpleField(name="sourcefile", type="Edm.String", filterable=True, facetable=True)
            ],
            semantic_settings=SemanticSettings(
                configurations=[SemanticConfiguration(
                    name='default',
                    prioritized_fields=PrioritizedFields(
                        title_field=None, prioritized_content_fields=[SemanticField(field_name='content')]))]),
                vector_search=VectorSearch(
                    algorithm_configurations=[
                        VectorSearchAlgorithmConfiguration(
                            name="default",
                            kind="hnsw",
                            hnsw_parameters=HnswParameters(metric="cosine")
                        )
                    ]
                )
            )
        print(f"Creating {index} search index")
        index_client.create_index(index)
    else:
        print(f"Search index {index} already exists")


def update_embeddings_in_batch(sections):
    batch_queue = []
    copy_s = []
    batch_response = {}
    token_count = 0
    for s in sections:
        token_count += calculate_tokens_emb_aoai(s["content"])
        if token_count <= SUPPORTED_BATCH_AOAI_MODEL[openaimodelname]['token_limit'] and len(batch_queue) < SUPPORTED_BATCH_AOAI_MODEL[openaimodelname]['max_batch_size']:
            batch_queue.append(s)
            copy_s.append(s)
        else:
            emb_responses = compute_embedding_in_batch([item["content"] for item in batch_queue])
            print(f"Batch Completed. Batch size  {len(batch_queue)} Token count {token_count}")
            for emb, item in zip(emb_responses, batch_queue):
                batch_response[item["id"]] = emb
            batch_queue = []
            batch_queue.append(s)
            token_count = calculate_tokens_emb_aoai(s["content"])

    if batch_queue:
        emb_responses = compute_embedding_in_batch([item["content"] for item in batch_queue])
        print(f"Batch Completed. Batch size  {len(batch_queue)} Token count {token_count}")
        for emb, item in zip(emb_responses, batch_queue):
            batch_response[item["id"]] = emb

    for s in copy_s:
        s["embedding"] = batch_response[s["id"]]
        yield s

def index_sections(filename, sections):
    print(f"Indexing sections from '{filename}' into search index '{index}'")
    search_client = SearchClient(endpoint=f"https://{searchservice}.search.windows.net/",
                                    index_name=index,
                                    credential=search_creds)
    i = 0
    batch = []
    for s in sections:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

def remove_from_index(filename):
    try:
        print(f"Removing sections from '{filename or '<all>'}' from search index '{index}'")
        search_client = SearchClient(endpoint=f"https://{searchservice}.search.windows.net/",
                                    index_name=index,
                                    credential=search_creds)
        while True:
            filter = None if filename is None else f"sourcefile eq '{pathlib.Path(filename).name}'"
            r = search_client.search("", filter=filter, top=1000, include_total_count=True)
            if r.get_count() == 0:
                break
            r = search_client.delete_documents(documents=[{ "id": d["id"] } for d in r])
            print(f"\tRemoved {len(r)} sections from index")
            # It can take a few seconds for search results to reflect changes, so wait a bit
            time.sleep(2)

    except Exception as e:
        print(f"\nIndex not available, creating new indexes...")
        
        
def refresh_openai_token():
    """
    Refresh OpenAI token every 5 minutes if using Azure AD credentials
    """
    if CACHE_KEY_TOKEN_TYPE in open_ai_token_cache and open_ai_token_cache[CACHE_KEY_TOKEN_TYPE] == 'azure_ad':
        if open_ai_token_cache[CACHE_KEY_CREATED_TIME] + 300 < time.time():
            token_cred = open_ai_token_cache[CACHE_KEY_TOKEN_CRED]
            if isinstance(token_cred, AzureDeveloperCliCredential):
                openai.api_key = token_cred.get_token("https://cognitiveservices.azure.com/.default").token
                open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
            else:
                print("The token credential object is not an instance of AzureDeveloperCliCredential.")
    else:
        # For 'azure' type, the API key does not need refreshing as it's static
        openai.api_key = openaikey
        

def read_files_from_adls(filename: str, use_vectors: bool, vectors_batch_support: bool, embedding_deployment: str):
    """
    Recursively read directory structure in an ADLS Gen 1 container
    and execute indexing for the individual files.
    """

    try:

        # Uploading spilt files for knowledge base 
        upload_blobs(filename)

        # Process the blob similar to a file
        page_map = get_document_text(filename)


        sections = create_sections(filename, page_map, use_vectors and not vectors_batch_support, embedding_deployment)

        if use_vectors and vectors_batch_support:
            sections = update_embeddings_in_batch(sections)
            
        index_sections(filename, sections)

    except Exception as e:
        print(f"\tGot an error while reading {filename} -> {e} --> skipping file")


    
# Main execution block
if __name__ == "__main__":

    # ADLS account details
    storageaccount_name = AZURE_STORAGE_NAME
    storageaccount_key = AZURE_ACCESS_KEY
    storageaccount_url = AZURE_STORAGE_URL
    upload_container_name = AZURE_UPLOAD_STORAGE_CONTAINER_NAME
    split_container_name = AZURE_SPLIT_STORAGE_CONTAINER_NAME
    openaideployment = AZURE_OPENAI_EMB_DEPLOYMENT
    openaimodelname = AZURE_OPENAI_EMB_MODEL
    index = AZURE_SEARCH_INDEX_NAME
    searchservice = AZURE_SEARCH_SERVICE_NAME
    search_creds = AzureKeyCredential(AZURE_SEARCH_SERVICE_ACCESS_KEY)
    openaiservice = AZURE_OPENAI_SERVICE_NAME
    openaikey = AZURE_OPENAI_SERVICE_ACCESS_KEY

    blob_service_client = BlobServiceClient(storageaccount_url, storageaccount_key)
    container_client = blob_service_client.get_container_client(upload_container_name)
    split_container_client = blob_service_client.get_container_client(split_container_name)
    blobs = container_client.list_blobs()
    azd_credential = AzureDeveloperCliCredential()

    use_vectors = True 
    vectors_batch_support = True  



    if use_vectors:
        if openaikey:
            openai.api_type = "azure"
            openai.api_key = openaikey
        else:

            open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
            open_ai_token_cache[CACHE_KEY_TOKEN_CRED] = azd_credential
            open_ai_token_cache[CACHE_KEY_TOKEN_TYPE] = "azure_ad"
            refresh_openai_token()


        openai.api_base = f"https://{openaiservice}.openai.azure.com"
        openai.api_version = "2022-12-01"
    
    # for processing each file in container
    for blob in blobs:
        filename = blob.name


        print("\nREMOVING EXISTING INDEX ....")
        remove_from_index(filename)
        create_search_index(index)
        print("Processing files...")
        read_files_from_adls(filename, use_vectors, vectors_batch_support, openaideployment)
        


