from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from llama_index.core import (
    load_index_from_storage,
    StorageContext,
)



# Load the storage context and index
SC_retrieved_sentence = StorageContext.from_defaults(persist_dir="./sentence_index")
retrieved_sentence_index = load_index_from_storage(SC_retrieved_sentence)

# Initialize the query engine
sentence_query_engine = retrieved_sentence_index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

# Define the template
template = """
    You are a helpful first aid assistant who has a lot of experience as an Emergency Medical Technician.
    
    You need to go through the knowledge base you have in order to provide the best possible advice to the user. 
    This is important: DO NOT GIVE FALSE INFORMATION IF YOU CAN NOT FIND ANY INFORMATION IN THE KNOWLEDGE BASE. 
    
    When you are providing the answer please follow this pattern:
        1. Do not say anything before and after the response.
        2. The response should only consist of the steps that the user should follow in giving first aid.
        3. The response should be in a JSON format. (e.g. "no_steps": 3, "step_1": "What to do as step 01", "step_2": "What to do as step 01", "step_3": "What to do as step 01")
        4. The response should be in a single line.
        
    So the emergency I have is {question}
"""

# Format the template with the question
formatted_query = template.format(question="i cut my finger while cooking what should i do?")

# Print the formatted query
print(formatted_query)

# Query the knowledge base
sentence_response = sentence_query_engine.query(formatted_query)

# Print the response
print(sentence_response)
