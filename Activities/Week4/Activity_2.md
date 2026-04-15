# Activity: Exploring the Gemini API (Part 1/2)

**Objective:** To collaboratively interact with a large language model (LLM) via an Application Programming Interface (API), we’ll use Google's models as our primary example. This approach enables us to leverage powerful, pre-trained models without managing the underlying infrastructure. We’ll explore the capabilities of the LLM API and demonstrate how to integrate its features into simple Gradio interfaces.

**Instructions for Breakout Rooms:**
*   Work together in your assigned group.
*   One member should share their screen with a Google Colab notebook environment ready.
*   Discuss each task, review the corresponding lecture material or demo notebook sections, and implement the code collaboratively.
*   The goal is understanding and experimentation, not just copying code. Discuss the results and any challenges encountered.
*   If you get stuck on a task, refer to the solution provided within the `<details>` tag. Try to solve it first before looking.

## Part 1: Setup and Initialization

This section ensures your environment is correctly configured to use the Gemini API. Follow these steps carefully.

### 1.1 Create an API key

If you haven't already, you need a Google API key to use Gemini.
You can [create](https://aistudio.google.com/app/apikey) your API key using Google AI Studio with a single click. Follow the instructions provided there.

**Important:** Treat your API key like a password. Do not share it publicly or commit it to version control systems like GitHub.

### 1.2 Add your key to Colab Secrets

Using Colab Secrets is the recommended way to handle your API key securely within Google Colab.

1.  Open your Google Colab notebook.
2.  Click on the **🔑 Secrets** tab in the left panel. (Refer to the image in the lecture notes if needed).
3.  Create a new secret. Enter the name `GOOGLE_API_KEY`.
4.  Paste the API key you created in step 1.1 into the `Value` input box.
5.  Ensure the **"Notebook access"** toggle button on the left is enabled (usually blue/on).

### 1.3 Install SDK and Initialize Client

Now, write and execute the Python code to install the necessary library and set up the API client using your stored secret key.

1.  **Install the SDK:** Run the following command in a code cell.
    ```python
    %pip install -q google-genai gradio
    ```
    *(Explanation: This installs the Google GenAI SDK and the Gradio library.)*

2.  **Import Libraries:** Run the following code in a *new* cell to import all the modules needed for this lab.
    ```python
    # Core Gemini and Colab libraries
    import google.generai as genai
    from google.generai import types
    from google.colab import userdata

    # Gradio for UI
    import gradio as gr

    # Utilities for display, file handling, JSON, etc.
    from IPython.display import Markdown, Image as IPImage
    import json
    from pydantic import BaseModel, Field # For structured JSON later
    from typing import Optional, List    # For type hinting later
    import requests                      # For downloading files
    import pathlib                       # For handling file paths
    from PIL import Image as PILImage    # For image manipulation
    import io                            # For handling byte streams
    import time                          # For delays (e.g., waiting for file processing)
    import base64                        # For potential image decoding

    print("Libraries imported successfully.")
    ```
    *(Explanation: This cell brings all the necessary Python tools into your notebook's memory so you can use their functions and classes.)*

### 1.4 Initialize API Client and Select Model

Retrieve your API key from Colab Secrets and use it to create the Gemini client object.

1.  **Retrieve Key and Initialize Client:** Run the following code in a *new* cell. This assumes your `GOOGLE_API_KEY` secret is correctly set up as per step 1.2.
    ```python
    # Retrieve the API key from Colab Secrets
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

    # Initialize the client - This line connects to Google's service
    # If the key is invalid or missing, this line will likely cause an error.
    client = genai.Client(api_key=GOOGLE_API_KEY)

    print("Gemini Client initialized.")

    # Choose a model ID to use for subsequent requests
    MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"] {"allow-input":true, isTemplate: true}
    print(f"Using Model ID: {MODEL_ID}")
    ```
    *(Explanation: This code gets your secret key, uses it to create the `client` object which is your main tool for talking to Gemini, and sets a default `MODEL_ID` variable.)*

    **Important Note:** If the cell above fails, double-check that:
    *   You completed step 1.2 correctly (Secret name is `GOOGLE_API_KEY`, value is correct, Notebook access is ON).
    *   You have internet connectivity.
    *   The API key itself is valid.

<details>
<summary>More Robust Initialization Code (Optional)</summary>

For situations where you want to handle potential errors during initialization more gracefully (e.g., if the key is missing or invalid), you could use a `try...except` block like this:

```python
# Robust Initialization Example:
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        # Raise an error specifically if the key wasn't found in secrets
        raise ValueError("API Key 'GOOGLE_API_KEY' not found in Colab Secrets. Please ensure it's set correctly and Notebook access is enabled.")

    # Initialize the client
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("Successfully initialized Gemini Client.")

    # Choose a model (only if client was initialized)
    MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"] {"allow-input":true, isTemplate: true}
    print(f"Using Model ID: {MODEL_ID}")

# Catch any exception during the process
except Exception as e:
    print(f"Error during initialization: {e}")
    print("\nPlease check:")
    print("- API key is correct in Google AI Studio / Cloud Console.")
    print("- Secret 'GOOGLE_API_KEY' exists in Colab Secrets with the correct value.")
    print("- 'Notebook access' is enabled for the secret.")
    print("- Internet connection is active.")
    # Prevent further code relying on 'client' from running if it failed
    # You might need to handle the absence of 'client' in later cells if using this block.
```

This robust version provides more specific feedback if something goes wrong, but the simpler version above is sufficient if you carefully follow the setup steps.

</details>

**Group Discussion:** Confirm that everyone understands the purpose of the API key, Colab Secrets, the SDK, and the client initialization. Ensure the client initializes successfully (you should see the "Gemini Client initialized" message) before proceeding to the next part of the lab. If there are errors, troubleshoot using the 'Important Note' points above.

## Part 2: Basic Text Generation

Explore sending simple text prompts and displaying responses.

### Task 2.1: Simple Text Prompt

1.  Write code to ask the Gemini model a factual question, such as "What is the capital of France?".
2.  Use the `client.models.generate_content` method with your `MODEL_ID` and the prompt.
3.  Print the `response.text`.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully in the previous step.
prompt = "What is the capital of Finland?"
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)
print(f"Prompt: {prompt}")
print(f"Response:\n{response.text}")
```

</details>

### Task 2.2: Text Prompt with Gradio

1.  Adapt the code from the lecture notebook to create a simple Gradio interface.
2.  The interface should have one `gr.Textbox` for input and one `gr.Markdown` for output.
3.  The function called by Gradio should take the user's prompt, call `generate_content`, and return the `response.text`.
4.  Test the interface with a few different prompts.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
def ask_gemini_gradio(user_prompt):
    if not user_prompt:
        return "Please enter a prompt."
    # Note: If API call fails here, Gradio might show a generic error.
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=user_prompt
    )
    return response.text

# Create Gradio interface
text_qa_interface = gr.Interface(
    fn=ask_gemini_gradio,
    inputs=gr.Textbox(lines=3, placeholder="Enter your question here...", label="Your Prompt"),
    outputs=gr.Markdown(label="Gemini Response"),
    title="Simple Gemini Q&A",
    description="Enter a prompt and get a text response from the Gemini model."
)
text_qa_interface.launch()
```

</details>

**Group Discussion:** Discuss the difference between running the code directly and using the Gradio interface. How does Gradio simplify interaction for a non-programmer?

## Part 3: Multimodal Input (Image + Text)

Explore sending prompts that include both images and text.

### Task 3.1: Image Analysis Prompt

1.  Find the URL of an image online (e.g., a picture of a specific animal or object).
2.  Write code to:
    *   Download the image using `requests`.
    *   Open the image using `PIL.Image`.
    *   Send a prompt to Gemini containing *both* the PIL image object and a text question about the image (e.g., "What type of animal is this?", "Describe the main object in this image.").
    *   Use a model that supports multimodal input (e.g., `gemini-2.5-flash`).
    *   Print the `response.text`.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Example Image URL (replace with your own if desired)
IMAGE_URL = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png" # Google Pixel phone

# Download and open image
img_bytes = requests.get(IMAGE_URL).content
pil_image = PILImage.open(io.BytesIO(img_bytes))

# Prepare prompt
text_prompt = "Describe the object shown in this image. What might it be used for?"

# Send multimodal request
response = client.models.generate_content(
    model=MODEL_ID, # Assumes MODEL_ID is multimodal capable
    contents=[pil_image, text_prompt] # List contains Image and Text
)

print(f"Text Prompt: {text_prompt}")
print(f"Response:\n{response.text}")

# Optionally display the image in Colab
# from IPython.display import display
# display(pil_image.resize((200,200)))

```

</details>

### Task 3.2: Image Analysis with Gradio

1.  Adapt the multimodal Gradio example from the lecture/notebook.
2.  The interface should take an uploaded image (`gr.Image(type="pil")`) and a text prompt (`gr.Textbox`).
3.  The function should send both the image (as a PIL object) and the text prompt to Gemini.
4.  The output should be the text response displayed in `gr.Markdown`.
5.  Test by uploading an image and asking a question about it.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
def analyze_image_gradio(image_input, text_prompt):
    if image_input is None:
        return "Please upload an image."
    if not text_prompt:
        return "Please enter a text prompt."

    # image_input is already a PIL image due to type="pil"
    pil_image = image_input

    # Note: If API call fails here, Gradio might show a generic error.
    response = client.models.generate_content(
        model=MODEL_ID, # Assumes multimodal capable model
        contents=[pil_image, text_prompt]
    )
    return response.text

# Create Gradio interface
image_analysis_interface = gr.Interface(
    fn=analyze_image_gradio,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(lines=2, placeholder="Ask something about the image...", label="Text Prompt")
    ],
    outputs=gr.Markdown(label="Analysis Response"),
    title="Image + Text Analysis with Gemini",
    description="Upload an image and provide a text prompt for Gemini to analyze."
)
image_analysis_interface.launch()

```

</details>

**Group Discussion:** Discuss potential applications for multimodal models that can understand both images and text.

## Part 4: Configuring Generation Parameters

Explore how changing parameters like `temperature` affects the model's output.

### Task 4.1: Experiment with Temperature

1.  Choose a creative prompt (e.g., "Write a short poem about a rainy day in Helsinki.").
2.  Call `client.models.generate_content` with this prompt **twice**:
    *   First time, set `config=types.GenerateContentConfig(temperature=0.1)`.
    *   Second time, set `config=types.GenerateContentConfig(temperature=0.9)`.
3.  Print both responses and compare them.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
creative_prompt = "Write a short poem about a rainy day in Helsinki."
print(f"Prompt: {creative_prompt}\n")

# Low temperature (more focused)
config_low_temp = types.GenerateContentConfig(temperature=0.1)
response_low = client.models.generate_content(
    model=MODEL_ID,
    contents=creative_prompt,
    config=config_low_temp
)
print(f"--- Response (Temperature: 0.1) ---\n{response_low.text}\n")

# High temperature (more creative/random)
config_high_temp = types.GenerateContentConfig(temperature=0.9)
response_high = client.models.generate_content(
    model=MODEL_ID,
    contents=creative_prompt,
    config=config_high_temp
)
print(f"--- Response (Temperature: 0.9) ---\n{response_high.text}\n")

```

</details>

### Task 4.2: Parameter Control with Gradio

1.  Use the Gradio example from the lecture/notebook that allows controlling `temperature`, `max_output_tokens`, and potentially `top_k` or `top_p` via sliders/number inputs.
2.  Run the interface.
3.  Experiment with a single prompt but adjust the parameters using the UI controls. Observe how the output changes.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Ensure the function definition and gr.Interface call are present and executable.
def generate_response_configured(prompt, temperature, max_tokens): # Simplified version
    if not prompt:
        return "Please enter a prompt."

    config = types.GenerateContentConfig(
        temperature=float(temperature),
        max_output_tokens=int(max_tokens)
    )
    # Note: If API call fails here, Gradio might show a generic error.
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config
    )
    return response.text

config_interface = gr.Interface(
    fn=generate_response_configured,
    inputs=[
        gr.Textbox(label="Prompt", lines=3),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Temperature"),
        gr.Number(value=150, label="Max Output Tokens", precision=0)
    ],
    outputs=gr.Markdown(label="Model Response"),
    title="Gemini Prompt with Config Control",
    description="Experiment with temperature and max tokens."
)
config_interface.launch()

# Note: The full example in the lecture included more parameters (top_k, top_p, etc.)
# You can use that more complete version if preferred.
```

</details>

**Group Discussion:** Discuss when you might want a lower temperature versus a higher temperature for different types of tasks (e.g., factual summary vs. brainstorming).

## Part 5: Multi-turn Chat

Explore maintaining conversation context.

### Task 5.1: Simple Chat Sequence

1.  Referencing the chat example in the lecture/notebook:
    *   Create a chat session using `client.chats.create()`. You can optionally add a `system_instruction` (e.g., "You are a helpful assistant.").
    *   Send an initial message using `chat.send_message()` (e.g., "What are the main steps to bake bread?"). Print the response.
    *   Send a follow-up message that relies on the previous context (e.g., "What kind of flour is best for the first step?"). Print the response.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Start chat
chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful baking assistant."
    )
)
print("Chat session started.\n")

# First message
msg1 = "What are the main steps to bake a simple loaf of bread?"
print(f"User: {msg1}")
response1 = chat.send_message(msg1)
print(f"Assistant:\n{response1.text}\n")

# Second message (contextual)
msg2 = "What kind of flour is generally recommended for the kneading step?"
print(f"User: {msg2}")
response2 = chat.send_message(msg2)
print(f"Assistant:\n{response2.text}\n")

```

</details>

**Group Discussion:** How does the `chat` object help maintain context? Discuss the limitation of the simple Gradio chat example provided in the lecture (it starts a new chat each time).