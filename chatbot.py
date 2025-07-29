from gradio_client import Client, handle_file
from config import GRADIO_API_URL, WARP_IMAGE_PATH
from audio import speech_to_text, text_to_speech


class ChatbotSession:
    """
    A session handler for interacting with the chatbot using both
    voice and text inputs. Supports maintaining chat history and
    returns text + audio responses.
    """

    def __init__(self):
        # Stores conversation history as a list of (question, answer) tuples
        self.chat_history = []

        # Initialize Gradio API client
        self.client = Client(GRADIO_API_URL)

    def get_chatbot_response(self, question, image_path):
        """
        Sends a question and image to the Gradio chatbot API and returns the response.
        """
        try:
            response = self.client.predict(handle_file(image_path), question)
            return response
        except Exception as e:
            print(f"Gradio API Error: {e}")
            return f"Error: Failed to get response from chatbot."

    def voice_chatbot(self, audio_file):
        """
        Processes a voice input:
        1. Converts speech to text.
        2. Gets chatbot's answer.
        3. Converts answer to audio.
        Returns updated chat history, answer text, and audio path.
        """
        # Convert user voice to text
        question = speech_to_text(audio_file)

        # If transcription failed
        if question.startswith("Error"):
            self.chat_history.append(("User (voice)", question))
            return self.chat_history, "", None

        # Get answer from chatbot
        answer = self.get_chatbot_response(question, WARP_IMAGE_PATH)

        # Add to chat history and synthesize audio
        self.chat_history.append((question, answer))
        audio_response_path = text_to_speech(answer)

        return self.chat_history, answer, audio_response_path

    def text_chatbot(self, user_text):
        """
        Processes a text input:
        1. Sends text to chatbot.
        2. Converts answer to audio.
        Returns updated chat history, answer text, and audio path.
        """
        # Directly use user input as question
        question = user_text

        # Get answer from chatbot
        answer = self.get_chatbot_response(question, WARP_IMAGE_PATH)

        # Add to chat history and synthesize audio
        self.chat_history.append((question, answer))
        audio_response_path = text_to_speech(answer)

        return self.chat_history, answer, audio_response_path
