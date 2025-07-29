import gradio as gr
import cv2
from vision import detect_painting, stream_live, VisionState
from chatbot import ChatbotSession
from feedback import save_feedback

# Initialize Gradio app
with gr.Blocks() as demo:

    # App header and description
    gr.Markdown(
        """
        <div style="text-align:center">
            <h1>Odin Vision</h1>
            <h3>
            OdinVision is an assistive technology system designed to help visually impaired individuals perceive and interpret visual artworks—particularly paintings—through a multisensory experience powered by real-time computer vision, depth mapping, and human-computer interaction techniques.
            </h3>
            <p>
            At its core, OdinVision captures a live video feed of the environment and uses advanced depth sensing to generate a real-time depth map of the scene. This allows the system to accurately estimate the spatial layout and proximity of objects, particularly flat surfaces like paintings. Once a painting is detected and the depth data is obtained, the system overlays this spatial information with color recognition and hand-tracking to create an interactive experience.
            </p>
        </div>
        """
    )

    # Display image outputs
    with gr.Row():
        image_output1 = gr.Image(label="Colour Overlay View", elem_id="image_output1")
        image_output2 = gr.Image(label="Depth Overlay View", elem_id="image_output2")

    # Painting detection controls
    detect_btn = gr.Button("Detect Painting")
    status_box = gr.Textbox(label="Status")
    state = VisionState()

    # Stream live video
    demo.load(
        fn=lambda: stream_live(state), inputs=[], outputs=[image_output1, image_output2]
    )

    # Detect painting on button click
    detect_btn.click(fn=lambda: detect_painting(state), inputs=[], outputs=status_box)

    # Chatbot UI section
    chatbot = gr.Chatbot(label="Painting Chatbot")

    # Voice input section
    gr.Markdown(
        """
        <h2>Ask by Voice</h2>
        <p>Use the button below to convey your questions to the chatbox verbally.</p>
    """
    )
    audio_input = gr.Audio(type="filepath", label="Voice Question")
    send_voice_btn = gr.Button("Send Voice Question")

    # Text input section
    gr.Markdown(
        """
        <h2>Ask by Text</h2>
        <p>Use the box below to convey your questions to the chatbox textually.</p>
    """
    )
    text_input = gr.Textbox(
        label="Text Question", placeholder="Type your question here..."
    )
    send_text_btn = gr.Button("Send Text Question")

    # Chatbot response outputs
    response_textbox = gr.Textbox(label="Textual Response", visible=False)
    audio_output = gr.Audio(label="Voice Response")

    # Initialize chatbot session
    chat_session = ChatbotSession()

    # Voice question handling
    send_voice_btn.click(
        fn=chat_session.voice_chatbot,
        inputs=audio_input,
        outputs=[chatbot, response_textbox, audio_output],
    )

    # Text question handling
    send_text_btn.click(
        fn=chat_session.text_chatbot,
        inputs=text_input,
        outputs=[chatbot, response_textbox, audio_output],
    )

    # Feedback form
    feedback_input = gr.Textbox(
        label="Feedback", placeholder="Type your feedback here..."
    )
    submit_feedback_btn = gr.Button("Submit Feedback")
    feedback_output = gr.Textbox(label="Feedback Status")

    # Save feedback to CSV on submit
    submit_feedback_btn.click(
        fn=lambda feedback: save_feedback(feedback),
        inputs=feedback_input,
        outputs=feedback_output,
    )

# Launch the Gradio app
demo.launch()
