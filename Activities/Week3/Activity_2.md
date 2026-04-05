# Introduction to Gradio: Building Machine Learning Interfaces

This document introduces Gradio, a Python library used to build web interfaces for machine learning models. 

This lab serves as a prerequisite for your upcoming projects. Throughout this guide, we will use "mock" (dummy) functions to represent machine learning models. These mock functions will accept various inputs—such as text, images, and audio—and return simulated outputs. 

Once you understand how to structure these interfaces, you will be able to adapt these examples for your own project prototypes. You will simply replace the mock functions with your actual, trained machine learning models or API calls (e.g., text generators, image classifiers, or audio transcribers).

This guide focuses on the core 80% of Gradio's functionality, which provides the foundation needed to build almost any standard machine learning prototype. 

---

## Part 0: Environment Setup

Gradio is an external Python package. When working in Google Colab, it is not pre-installed, meaning you must run an installation command at the beginning of every new session.

> [!IMPORTANT]  
> If you consult the official Gradio documentation or tutorials, they frequently recommend using the command `pip install --upgrade gradio` or `pip install -U gradio`. **Do not use the upgrade flag in Google Colab.**
> 
> Google Colab relies on specific, pre-configured versions of underlying Python packages to keep the notebook environment running correctly. The `--upgrade` command forces Python to update Gradio and all of its core dependencies to their absolute newest versions. This frequently creates version conflicts with Colab's built-in libraries, breaks the notebook connection, and prevents your interface from loading. 

**Run the following command in your very first code cell:**

```python
!pip install gradio -q
```

<!-- 
```python
#!pip install gradio==5.50.0  -q
# print(gr.__version__)
``` 
-->


<details>
<summary><strong>What do the <code>!</code> and <code>-q</code> mean?</strong></summary>
<ul>
<li><strong><code>!</code> (Exclamation Mark):</strong> This tells Google Colab to run the line as a system terminal command, rather than as standard Python code.</li>
<li><strong><code>-q</code> (Quiet Flag):</strong> Installing a large library like Gradio normally outputs hundreds of lines of text as it downloads various dependencies. Appending <code>-q</code> suppresses this terminal output, keeping your notebook visually clean and readable.</li>
</ul>
</details>




---

## Part 1A: The `Interface` Class

The `gr.Interface` class is the standard, fastest method for building a machine learning prototype. It automatically generates a web page with a clean layout: user inputs are placed on the left, model outputs are placed on the right, and a "Submit" button is added at the bottom.

To build an `Interface`, you must provide three core components:
1. **`fn`**: The Python function that represents your machine learning model (the "brain").
2. **`inputs`**: The type of data the user will provide (the UI components on the left).
3. **`outputs`**: The type of data the interface will return (the UI components on the right).


### Example 1.1: Basic Text Generation
This is the simplest form of an interface. It accepts a text prompt from the user, processes it through a mock function, and returns a text response.

```python
import gradio as gr

# 1. Define the mock function
def text_generation_mock(prompt):
    return "This is a simulated AI response to: " + prompt

# 2. Configure the interface
demo = gr.Interface(
    fn=text_generation_mock, 
    inputs="text", 
    outputs="text",
    title="Basic Text Prototype"
)

# 3. Launch the interface
demo.launch() 
```

<!-- 
> [!TIP]  
> **Running Interfaces in Google Colab:** If your interface looks compressed or does not load properly inside the Colab cell, change the last line of your code to `demo.launch(share=True)`. This generates a secure, public web link (valid for 72 hours) that allows you to open your prototype in a full browser tab or share it with others. 
-->



> [!NOTE]  
> Notice that when assigning the function to the interface (`fn=text_generation_mock`), you provide the name of the function **without parentheses `()`**. You are passing the function itself to Gradio, so Gradio can call it later when the user clicks the Submit button.


### Example 1.2: Multiple Inputs (Playground Style)
Real machine learning models often require multiple parameters, such as a text prompt, a numerical setting (like "Temperature" or "Creativity"), and categorical choices. 

If your Python function requires multiple arguments, you must pass a **Python list `[]`** to the `inputs` parameter.

```python
import gradio as gr

def advanced_generation_mock(prompt, temperature, model_choice):
    # This mock function simulates processing multiple settings
    result = f"Prompt: {prompt}\n"
    result += f"Model Used: {model_choice}\n"
    result += f"Creativity Level (Temperature): {temperature}\n"
    return result

demo = gr.Interface(
    fn=advanced_generation_mock,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your prompt here...", label="User Prompt"), 
        gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="Temperature"),
        gr.Dropdown(["Standard Model", "Pro Model", "Fast Model"], value="Standard Model", label="Select AI Model")
    ],
    outputs=gr.Textbox(lines=3, label="System Output"),
    title="LLM Playground"
)

demo.launch()
```


> [!IMPORTANT]  
> **Order Matters:** The order of the components in your `inputs` list must exactly match the order of the arguments in your Python function. In the example above: 
> 1. `Textbox` maps to `prompt`
> 2. `Slider` maps to `temperature`
> 3. `Dropdown` maps to `model_choice`
> 4. In Gradio, there isn’t a separate Textarea component anymore. gr.Textbox becomes a textarea when you increase the lines parameter e.g `gr.Textbox(lines=5,..`



### Example 1.3: Image Inputs (Computer Vision)
For image classification or object detection projects, Gradio provides a `gr.Image` component. It automatically handles uploading files from a computer or capturing images directly from a webcam.

```python
import gradio as gr

def image_classification_mock(image_path):
    # In a real project, this function passes the image file to a vision model.
    # Here, we return dummy confidence scores for demonstration.
    return {"Dog": 0.88, "Cat": 0.10, "Bird": 0.02} 

demo = gr.Interface(
    fn=image_classification_mock,
    inputs=gr.Image(type="filepath", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3), 
    title="Vision Classifier"
)

demo.launch()
```

<details>
<summary><strong>Why do we use <code>type="filepath"</code> for media?</strong></summary>
By default, Gradio converts uploaded images into complex mathematical arrays (NumPy arrays) so they can be processed by local PyTorch or TensorFlow models. 
<br><br>
However, if you are building a prototype that sends data to an external API (like OpenAI, Gemini, or Hugging Face), handling NumPy arrays is extremely difficult. Setting <code>type="filepath"</code> tells Gradio to save the uploaded image to the Colab environment and simply pass the file location (a string like <code>"/tmp/gradio/image.png"</code>) to your Python function. Passing a file path to an API is standard practice and much easier to code.
</details>


### Example 1.4: Audio Inputs (Speech-to-Text)
If your prototype involves speech recognition or audio analysis, you can use the `gr.Audio` component. You can configure it to allow live microphone recording, file uploads, or both.

```python
import gradio as gr

def speech_to_text_mock(audio_path):
    # This mock function pretends to transcribe the audio file
    if audio_path is None:
        return "Error: No audio provided."
    
    return f"Successfully received audio file at: {audio_path}. Transcription: 'Hello, this is a test recording.'"

demo = gr.Interface(
    fn=speech_to_text_mock,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio"),
    outputs=gr.Textbox(lines=10, label="Transcription Result"),
    title="Speech Recognition Prototype"
)

demo.launch()
```

----

## Part 1B: Knowledge Check (The `Interface` Class)

Test your understanding of the concepts covered in Part 1. Read the questions carefully and try to answer them before clicking to reveal the solution.

### Question 1: Function Syntax
When setting up a `gr.Interface`, why do we write `fn=text_generation_mock` instead of `fn=text_generation_mock()`?

<details>
<summary><strong>View Answer</strong></summary>

If you include the parentheses `()`, Python executes the function immediately when the code cell runs, before the interface even loads. 

By omitting the parentheses, you are passing the *reference* of the function to Gradio. This tells Gradio: *"Here is the function. Please hold onto it and only run it when the user clicks the Submit button."*
</details>

<br>

### Question 2: Handling Multiple Inputs
Imagine you have written the following Python function for a housing price predictor:
```python
def predict_price(square_footage, num_bedrooms, has_pool):
    # logic here...
    return estimated_price
```
How would you write the `inputs` parameter inside `gr.Interface` to match this function correctly?

<details>
<summary><strong>View Answer</strong></summary>

You must provide a Python list containing three Gradio components, and they must be in the exact same order as the function arguments.

```python
inputs=[
    gr.Number(label="Square Footage"),    # Maps to square_footage
    gr.Slider(1, 10, step=1, label="Bedrooms"), # Maps to num_bedrooms
    gr.Checkbox(label="Has Pool?")        # Maps to has_pool
]
```
</details>

<br>

### Question 3: Debugging an Interface
The following code will crash when the user clicks submit. Can you identify why?

```python
import gradio as gr

def summarize_text(article_text):
    return "Summary: " + article_text[:50]

demo = gr.Interface(
    fn=summarize_text,
    inputs=[gr.Textbox(label="Article"), gr.Slider(label="Creativity")],
    outputs="text"
)
demo.launch()
```

<details>
<summary><strong>View Answer</strong></summary>

**Mismatch between inputs and function arguments.** 

The `inputs` list has two components (a `Textbox` and a `Slider`). This means Gradio will try to send *two* pieces of data to the function. However, the `summarize_text` function only accepts *one* argument (`article_text`). 

To fix this, you either need to remove the `Slider` from the UI, or update the Python function to accept a second parameter (e.g., `def summarize_text(article_text, creativity):`).
</details>



---

## Part 2A: `Blocks` (Custom Layouts)

The `Interface` class you learned in Part 1 is incredibly fast, but it is rigid. It forces your app into a strict "inputs on the left, outputs on the right" design. 

If you are building a more complex prototype—such as a dashboard with side-by-side split screens, multiple tabs, or specific button placements—you must use the **`Blocks`** API. 

Think of `Blocks` as a blank canvas. You become the architect, placing UI components exactly where you want them.


### The Architects: Rows and Columns

To design a layout in Gradio, you use two fundamental container tools. You nest these containers inside each other using Python's `with` statement:

*   **`with gr.Row():`** Places everything inside it horizontally (side-by-side).
*   **`with gr.Column():`** Places everything inside it vertically (top-to-bottom).

**The Standard Dashboard Formula:** 
To create a classic split-screen layout, you create one `Row`, and put two `Column`s inside it. 

> [!IMPORTANT]  
> **The "Wiring" Rule:** In the `Interface` class, Gradio automatically connected your inputs and outputs. In `Blocks`, **nothing happens automatically**. You must manually connect a button to your Python function using an event listener, such as `.click()`.

### Example 2.1: The Split-Screen Dashboard
This example demonstrates how to build a layout with inputs on the left and outputs on the right, and how to wire the "Run" button to the mock function.

```python
import gradio as gr

# 1. Define the mock function
def text_summarizer_mock(text):
    if not text:
        return "Please enter text to summarize."
    # Simulate a summary by returning the first 30 characters
    return f"SUMMARY: {text[:30]}..."

# 2. Design the canvas
with gr.Blocks() as demo:
    gr.Markdown("## Document Summarization Dashboard") # Markdown for titles
    
    # Create a horizontal row
    with gr.Row():
        
        # Left side (Inputs)
        with gr.Column():
            input_text = gr.Textbox(lines=6, label="Paste Article Here")
            run_btn = gr.Button("Summarize Now", variant="primary") # 'primary' makes the button blue
            
        # Right side (Outputs)
        with gr.Column():
            output_text = gr.Textbox(label="AI Summary Results", lines=6)
            
    # 3. Wire the button to the function
    run_btn.click(
        fn=text_summarizer_mock, 
        inputs=input_text, 
        outputs=output_text
    )

demo.launch()
```

<details>
<summary><strong>Understanding the Event Listener: <code>.click()</code></strong></summary>
When you write <code>run_btn.click(...)</code>, you are telling Python:
<ol>
<li>Wait for the user to click the <code>run_btn</code>.</li>
<li>When clicked, grab the current text inside <code>input_text</code>.</li>
<li>Send that text into the <code>text_summarizer_mock</code> function.</li>
<li>Take whatever the function <code>returns</code>, and display it inside <code>output_text</code>.</li>
</ol>
</details>


### Example 2.2: Image Filtering Playground
This example mixes media (images) and parameters (dropdowns) in a custom layout. It proves that you can put settings right next to the image they affect.

```python
import gradio as gr
import os

def apply_filter_mock(image_path, filter_type):
    if image_path is None:
        return "No image uploaded.", "Error: No image uploaded."
    
    # Extract some info about the file
    file_name = os.path.basename(image_path)
    file_size = os.path.getsize(image_path)

    result_text = f"""
File Name: {file_name}
File Size: {file_size} bytes
Selected Filter: {filter_type}
"""

    status_message = f"Successfully processed '{filter_type}' filter."

    return result_text, status_message

with gr.Blocks() as demo:
    gr.Markdown("## Image Processing Playground (Text Mode)")
    
    with gr.Row():
        # Input Column
        with gr.Column():
            img_in = gr.Image(type="filepath", label="1. Upload Photo")
            filter_dropdown = gr.Dropdown(
                ["Black & White", "Sepia", "Blur"],
                value="Black & White",
                label="2. Select Filter"
            )
            apply_btn = gr.Button("3. Submit")
            
        # Output Column
        with gr.Column():
            text_output = gr.Textbox(label="Processed Info", lines=8)
            status_text = gr.Textbox(label="System Status")

    apply_btn.click(
        fn=apply_filter_mock, 
        inputs=[img_in, filter_dropdown], 
        outputs=[text_output, status_text]
    )

demo.launch()
```

<!-- 
```python
import gradio as gr

def apply_filter_mock(image_path, filter_type):
    if image_path is None:
        return None, "Error: No image uploaded."
    
    # In a real app, you process the image array here.
    # For this mock, we just return the original image path to display it on the screen.
    status_message = f"Successfully applied '{filter_type}' filter."
    return image_path, status_message

with gr.Blocks() as demo:
    gr.Markdown("## Image Processing Playground")
    
    with gr.Row():
        # Input Column
        with gr.Column():
            img_in = gr.Image(type="filepath", label="1. Upload Photo")
            filter_dropdown = gr.Dropdown(["Black & White", "Sepia", "Blur"], value="Black & White", label="2. Select Filter")
            apply_btn = gr.Button("3. Apply Filter")
            
        # Output Column
        with gr.Column():
            img_out = gr.Image(label="Processed Result")
            status_text = gr.Textbox(label="System Status")

    # Wire it together. Note that inputs and outputs are LISTS because there are multiple!
    apply_btn.click(
        fn=apply_filter_mock, 
        inputs=[img_in, filter_dropdown], 
        outputs=[img_out, status_text]
    )

demo.launch()
``` 
-->

> [!NOTE]  
> **Order is Critical:** Just like in Part 1, the order of components in the `inputs=[]` list must match the arguments in your Python function. Furthermore, the order of components in the `outputs=[]` list must match the order of the variables your Python function `returns`.


### Example 2.3: Conversational Chatbot layout
If you are building an AI agent (like a customer service bot), Gradio has a special `gr.Chatbot` component. This component expects a specific format: a list of message pairs (User Message, Bot Response).

```python
import gradio as gr

def chat_logic_mock(user_message, history):
    # 'history' automatically tracks previous messages
    bot_response = f"I am a mock AI. You just said: '{user_message}'."
    
    # Append the new interaction to the conversation history
    history.append((user_message, bot_response))
    
    # Return an empty string to clear the user's input box, and return the updated history
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("## Customer Support Agent")
    
    # The Chatbot component visualizes the history
    chatbot_ui = gr.Chatbot(label="Conversation History")
    
    with gr.Row():
        user_input = gr.Textbox(scale=4, placeholder="Type your message here...", label="Your Message")
        send_btn = gr.Button("Send", scale=1)
        clear_btn = gr.Button("Clear Chat", scale=1)

    # .submit() triggers when the user presses the "Enter" key on their keyboard
    user_input.submit(
        fn=chat_logic_mock, 
        inputs=[user_input, chatbot_ui], 
        outputs=[user_input, chatbot_ui]
    )
    
    # .click() triggers when the user clicks the "Send" button
    send_btn.click(
        fn=chat_logic_mock, 
        inputs=[user_input, chatbot_ui], 
        outputs=[user_input, chatbot_ui]
    )
    
    # Clear history function (lambda returns an empty list to the chatbot)
    clear_btn.click(lambda: None, None, chatbot_ui, queue=False)

demo.launch()
```

> [!TIP]  
> Notice the `scale=` parameter inside the `Textbox` and `Button` components. `scale` determines how much horizontal space a component takes up relative to others in the same Row. By setting the text box to `scale=4` and buttons to `scale=1`, the text box becomes much wider.

### Example 2.4: The Multi-Modal Dashboard (Putting it all together)
This final template proves you can combine anything. It takes an Image, Text, and Dropdown, and outputs both Text and a downloadable File. You can use this as a starting point for complex ML projects.

```python
import gradio as gr

def complex_processing_mock(image_path, instruction, mode):
    # Mocking the processing logic
    status = "Image uploaded successfully." if image_path else "No image provided."
    report_text = f"Analysis Mode: {mode}\nInstruction: {instruction}\nStatus: {status}\n\nResult: No anomalies detected."
    
    # Generate a dummy text file to simulate an exportable report
    file_path = "output_report.txt"
    with open(file_path, "w") as f:
        f.write(report_text)
        
    return report_text, file_path

with gr.Blocks() as demo:
    gr.Markdown("## 🌌 Multi-Modal Analysis Prototype")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Data")
            img_in = gr.Image(type="filepath", label="Upload Reference Image")
            text_in = gr.Textbox(lines=2, label="Specific Instructions")
            mode_in = gr.Dropdown(["Standard Analysis", "Deep Scan"], value="Standard Analysis", label="Mode")
            run_btn = gr.Button("Execute Analysis", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Results")
            text_out = gr.Textbox(label="Analysis Details", lines=7)
            file_out = gr.File(label="Download Full Report")

    run_btn.click(
        fn=complex_processing_mock,
        inputs=[img_in, text_in, mode_in], 
        outputs=[text_out, file_out]       
    )

demo.launch()
```


---


# Part 2B: Knowledge Check (`Blocks` & Layouts)

Test your understanding of the advanced layouts and event listeners covered in Part 2.

### Question 1: Rows vs. Columns
If you want to create a layout with three text boxes side-by-side (left, center, and right), how would you structure your `Row` and `Column` blocks?

<details>
<summary><strong>View Answer</strong></summary>

You would create one `Row`, and place three `Column`s inside it. 

```python
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            box_1 = gr.Textbox(label="Left")
        with gr.Column():
            box_2 = gr.Textbox(label="Center")
        with gr.Column():
            box_3 = gr.Textbox(label="Right")
```
</details>

<br>

### Question 2: The "Wiring" Order
Look carefully at the `run_btn.click(...)` method at the bottom of this code snippet:

```python
with gr.Blocks() as demo:
    gr.Markdown("## 🌌 Multi-Modal Analysis Prototype")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="filepath", label="Upload Reference Image")
            text_in = gr.Textbox(lines=2, label="Specific Instructions")
            mode_in = gr.Dropdown(["Standard Analysis", "Deep Scan"], value="Standard Analysis")
            run_btn = gr.Button("Execute Analysis")
            
        with gr.Column():
            text_out = gr.Textbox(label="Analysis Details")
            file_out = gr.File(label="Download Full Report")

    # THE WIRING:
    run_btn.click(
        fn=complex_processing_mock,
        inputs=[img_in, text_in, mode_in], 
        outputs=[text_out, file_out]       
    )
```

**Question:** What would happen if a developer accidentally changed the wiring inputs to `inputs=[text_in, img_in, mode_in]`?

<details>
<summary><strong>View Answer</strong></summary>

The data sent to the Python function `complex_processing_mock` would be backwards. 

The function would receive the user's text string where it expects an image file path, and it would receive an image file path where it expects text. This would immediately cause the Python function to crash or throw an error when it tries to process the data. 

**Rule of thumb:** The order of variables inside the `.click(inputs=[...])` list MUST exactly match the order of arguments in your `def` function.
</details>

<br>

### Question 3: Modifying an Existing Layout
Using the same code snippet from Question 2, imagine you want to add a "Clear Data" button right next to the "Execute Analysis" button (so they share the same horizontal space inside the left column). How would you change the code?

<details>
<summary><strong>View Answer</strong></summary>

To place two buttons side-by-side inside the left column, you need to nest a new `gr.Row()` specifically for the buttons.

```python
        with gr.Column():
            img_in = gr.Image(type="filepath", label="Upload Reference Image")
            text_in = gr.Textbox(lines=2, label="Specific Instructions")
            mode_in = gr.Dropdown(["Standard Analysis", "Deep Scan"])
            
            # Added a nested Row just for the buttons
            with gr.Row(): 
                run_btn = gr.Button("Execute Analysis", variant="primary")
                clear_btn = gr.Button("Clear Data")
```
</details>

<br>

### Question 4: Data Types in Blocks
When a user uploads a `.jpg` photo to the `img_in = gr.Image(type="filepath")` component, what exactly does the `complex_processing_mock` function receive in Python? Does it receive the raw image pixels, or something else?

<details>
<summary><strong>View Answer</strong></summary>

Because we specified `type="filepath"`, the Python function does **not** receive the raw image pixels or a complex array. 

Instead, it receives a **String** representing the location where Gradio temporarily saved the file (for example: `"/tmp/gradio/uploaded_image.jpg"`). This makes it incredibly easy to take that string and forward it to external machine learning APIs.
</details>

---
# Part 3: Using AI Assistants to Build Gradio UIs

Google Colab includes a built-in AI assistant (Gemini), and you can use it to write Gradio code for you. However, AI models often write outdated Gradio code or create messy layouts if your instructions are not highly specific. 

This section demonstrates how to correctly prompt an AI assistant to build a robust, custom interface, using an **AI Fitness Coach** as a concrete example.

### The Strategy: Iterative Prompting
Do not ask the AI to build the entire, fully-functional app (UI + real machine learning model) in one prompt. This usually results in broken code. Instead, use a step-by-step approach:
1. Generate the UI layout.
2. Refine the layout (Rows and Columns).
3. Generate the "Mock" (dummy) function and wire it together.
4. Replace the mock function with the real AI model.


### Step 1: The Main Prompt (The Skeleton)

Your first prompt should focus entirely on defining the inputs, the outputs, and requesting the `gr.Blocks` format. 

**Open the Gemini chat in Colab and use this prompt:**

> **User Prompt 1:**
> *"I am building a machine learning prototype in Google Colab using Gradio. I want to build a UI for an 'AI Fitness Coach'. Please generate the Gradio code using `gr.Blocks()`. Do not write the real AI logic yet. Just create the UI components.*
> 
> *Here is what the app needs:*
> *- Inputs: A number for 'Age', a dropdown for 'Fitness Goal' (Lose Weight, Build Muscle, Endurance), and a slider for 'Days per week' (1 to 7).*
> *- A 'Generate Plan' button.*
> *- Output: A large text box to display the 'Workout Plan'.*
> *- Keep it simple and use standard Gradio components."*

**What to expect:** Gemini will generate a functional, but likely vertical (top-to-bottom) list of components inside a `gr.Blocks()` container. 


### Step 2: The First Follow-up (Fixing the Layout)

Once you have the basic components, the next step is to format them into a professional layout. Tell the AI exactly how to structure the screen using Rows and Columns.

> **User Prompt 2:**
> *"This looks good, but the layout is entirely vertical. Please update the `gr.Blocks()` code to use a split-screen layout. Use `with gr.Row():` and `with gr.Column():`.*
> 
> *Put all the inputs (Age, Fitness Goal, Days per week) and the 'Generate Plan' button inside the left column. Put the output text box inside the right column. Add a Markdown title at the very top of the app."*

**What to expect:** Gemini will nest the components properly. Your code will now look similar to the professional dashboard examples covered in Part 2.


### Step 3: The Second Follow-up (Wiring and Mocking)

Now that the UI looks correct, it needs a "brain" so you can test it. Ask the AI to write a dummy function and connect the button.

> **User Prompt 3:**
> *"Now, please write a mock Python function called `generate_mock_plan(age, goal, days)`. The function should return a string that says: 'Based on your age of [age], your goal to [goal], and working out [days] days a week, here is a mock workout plan.'*
> 
> *After writing the function, update the Gradio code to wire the 'Generate Plan' button to this function using `.click()`. Ensure the `inputs` list matches the function arguments exactly."*

> [!TIP]  
> **Test the Code Now:** At this stage, you should copy the code Gemini provides, paste it into a Colab cell, and run it. You now have a fully functional, interactive UI. Ensure everything works before moving to the final step.


### Step 4: The Final Follow-up (Preparing for the Real LLM)

Once your prototype UI is working with the mock function, you are ready to connect it to a real Large Language Model (like the Gemini API or OpenAI API). 

To do this, the AI needs to take all the separate UI inputs and combine them into a single "Prompt String" that an LLM can understand. 

> **User Prompt 4:**
> *"The UI works perfectly. Now, I am ready to replace the mock function with a real call to an LLM API. *
> 
> *Please rewrite only the Python function. Rename it to `create_llm_prompt(age, goal, days)`. Inside the function, take those three variables and format them into a single, detailed text prompt that I can eventually send to a real AI model. Return that constructed prompt string."*

**The AI will provide a function similar to this:**

```python
def create_llm_prompt(age, goal, days):
    # This string formats the UI data into instructions for a real AI
    system_instructions = "You are an expert personal trainer."
    user_data = f"The user is {age} years old. Their main goal is to {goal}. They can commit to {days} days per week."
    task = "Please generate a detailed, day-by-day workout schedule based on this information."
    
    # Combine everything
    final_prompt = f"{system_instructions}\n\n{user_data}\n\n{task}"
    
    # In the final version of your project, you will pass 'final_prompt' to 
    # the real AI API here, and return the AI's response instead.
    return final_prompt
```

> [!NOTE]  
> **Why this method works:** By breaking the process down, you ensure that the UI is completely stable before introducing the complexity of real machine learning APIs. When errors occur, you will know exactly whether the error is coming from the Gradio UI or from the external API call.

---

## Reference: Component Data Types

When you are ready to replace these mock functions with your real machine learning models, it is crucial to know what data type Gradio is actually sending to your Python function.

Use this cheat sheet to understand how to handle the `inputs` in your code:

| Gradio Component | Python Data Type Received | Typical Use Case in ML Projects |
| :--- | :--- | :--- |
| `gr.Textbox()` | `str` (String) | Passing prompts to Language Models (LLMs). |
| `gr.Number()` | `int` or `float` | Exact numerical inputs (e.g., predicting age, housing prices). |
| `gr.Slider()` | `int` or `float` | Parameter tuning (e.g., AI creativity, confidence thresholds). |
| `gr.Checkbox()` | `bool` (True/False) | Boolean toggles (e.g., enabling safe-search). |
| `gr.Dropdown()` | `str` | Pre-defined selections (e.g., picking which model to run). |
| `gr.Image(type="filepath")` | `str` (File Path) | Passing image locations to Computer Vision models. |
| `gr.Audio(type="filepath")` | `str` (File Path) | Passing audio files to Speech-to-Text models (Whisper). |
| `gr.File()` | `str` (File Path) | Parsing external documents (PDF, CSV) for data analysis. |

---

### Beyond the 80%: Official Documentation

The examples provided in Part 1 and Part 2 cover the foundational patterns required to build the vast majority of machine learning prototypes. 

If your final project requires advanced features—such as custom CSS styling, dynamic component updates (e.g., making a slider appear only if a checkbox is checked), or managing complex server queues for heavy models—you should refer to the official Gradio documentation:

> **[Gradio Official Documentation](https://www.gradio.app/docs/)**