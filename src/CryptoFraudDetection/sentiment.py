import os
from typing import Callable

from dotenv import load_dotenv
import openai

# Read credentials from .env file
load_dotenv()

def get_client():
    """Creates an Azure OpenAI client instance.

    Returns:
        An Azure OpenAI client instance.
    """
    return openai.AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

def get_chat_bot(
    client: openai.Client | None = None,
    model: str = "gpt-4",
    content: str = "You are a sentiment analyser.",
) -> Callable[[str], str]:
    """Creates a chatbot for generating responses based on user input.

    Args:
        client: An OpenAI client instance. If not provided, a new client is created.
        model: The chat model to use.
        content: The system role description for the chatbot.

    Returns:
        A function that takes a prompt and returns the chatbot's response.
    """
    if not client:
        client = get_client()

    def create_response(prompt: str) -> str:
        """Generates a response for the given prompt using the chatbot.

        Args:
            prompt: The user-provided input prompt.

        Returns:
            The chatbot's response.

        Raises:
            ValueError: If the prompt exceeds the maximum token limit.
        """
        # GPT-4 token limit is approximately 8192 tokens, and we assume ~4 characters per token.
        if len(prompt) // 4 > 8192:
            raise ValueError("The prompt is too long. Please provide a shorter prompt.")

        # Call the OpenAI API to generate a chat completion
        response = client.chat.completions.create(
            messages=[
                # Define the assistant's behavior
                {
                    "role": "system",
                    "content": content
                },
                # User-provided input
                {"role": "user", "content": prompt}
            ],
            model=model
        )
        return response.choices[0].message.content

    return create_response

def sentiment(texts:list[str], chat_bot: Callable[[str], str] | None = None, max_score: int = 10) -> list[float]:
    """Evaluates the sentiment of each text using the chatbot and assigns a score.

    Args:
        texts: A list of texts to evaluate.
        chat_bot: A chatbot function to generate sentiment scores. If not provided, a new chatbot is created.
        max_score: The maximum score on the sentiment evaluation scale.

    Returns:
        A list of sentiment scores. Each score is an integer between 0 and max_score, or None if evaluation fails.
    """
    
    if not chat_bot:
        chat_bot = get_chat_bot()

    sentiment_scores = []
    for text in texts:
        chat_bot_query = (
            f"Rate the sentiment of the following text on a scale from 0 to {max_score}.\n"
            f"Text: {text}\n"
            f"Only give the integer value, DO NOT use words and Characters!"
        )
        response = chat_bot(chat_bot_query)
        if response:
            sentiment_scores.append(int(response))
        else:
            sentiment_scores.append(None)  # failure
    
    return sentiment_scores
