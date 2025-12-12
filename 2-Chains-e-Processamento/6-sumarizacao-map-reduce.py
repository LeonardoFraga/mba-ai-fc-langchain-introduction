from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

long_text = """
Dawn threads a pale gold through the alley of glass.
The city yawns in a chorus of brakes and distant sirens.
Windows blink awake, one by one, like sleepy eyes.
Streetcloth of steam curls from manholes, a quiet river.
Coffee steam spirals above a newspaper's pale print.
Pedestrians sketch light on sidewalks, hurried, loud with umbrellas.
Buses swallow the morning with their loud yawns.
A sparrow perches on a steel beam, surveying the grid.
The subway sighs somewhere underground, a heartbeat rising.
Neon still glows in the corners where night refused to retire.
A cyclist cuts through the chorus, bright with chrome and momentum.
The city clears its throat, the air turning a little less electric.
Shoes hiss on concrete, a thousand small verbs of arriving.
Dawn keeps its promises in the quiet rhythm of a waking metropolis.
The morning light cascades through towering windows of steel and glass,
casting geometric shadows on busy streets below.
Traffic flows like rivers of metal and light,
while pedestrians weave through crosswalks with purpose.
Coffee shops exhale warmth and the aroma of fresh bread,
as commuters clutch their cups like talismans against the cold.
Street vendors call out in a symphony of languages,
their voices mixing with the distant hum of construction.
Pigeons dance between the feet of hurried workers,
finding crumbs of breakfast pastries on concrete sidewalks.
The city breathes in rhythm with a million heartbeats,
each person carrying dreams and deadlines in equal measure.
Skyscrapers reach toward clouds that drift like cotton,
while far below, subway trains rumble through tunnels.
This urban orchestra plays from dawn until dusk,
a endless song of ambition, struggle, and hope.
"""

# Initialize a text splitter to divide the long text into smaller chunks
# Each chunk will have a maximum size of 250 characters with an overlap of 70 characters
# The overlap ensures that important context is preserved between chunks,
# which can be beneficial for processing by language models.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=70, )

# Split the long text into smaller parts (documents) using the text splitter
# Because chains like load_summarize_chain expect a list of Document objects as input.
parts = splitter.create_documents([long_text])

# Initialize a chat model (LLM) using Google's Gemini model
# The model is specified as "gemini-2.5-flash" and the provider is "google_genai"
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Load a summarization chain
# This chain uses the initialized LLM to summarize the input documents
# The chain type is "map_reduce", which first maps over the input documents
# to generate intermediate summaries and then reduces those summaries into a final output.
chain_sumarize = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)

# Invoke the summarization chain with the input documents
# The chain processes the documents and generates a summarized output
result = chain_sumarize.invoke({"input_documents": parts})

# Print the summarized output text
print(result["output_text"])