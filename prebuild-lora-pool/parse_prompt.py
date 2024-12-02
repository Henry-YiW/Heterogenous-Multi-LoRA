# Individual test file for parse_prompt.

from ollama import chat
from ollama import ChatResponse

# **** ollama interface ****

# from ollama import chat
# from ollama import ChatResponse

# response: ChatResponse = chat(model='phi3', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

def parse_prompt(user_prompt, max_keywords=5):
    """
    Parses the input prompt into a list of keywords using the Ollama LLaMA API.

    Args:
        user_prompt (str): The text prompt to parse.
        max_keywords (int): Maximum number of keywords to extract.

    Returns:
        list: A list of keywords extracted from the prompt.
    """
    # Define the prompt text
    llama_prompt = f"""
You are tasked with parsing a user-provided prompt designed for text-to-image generation. The goal is to extract meaningful *keywords* that represent the essential elements of the input prompt, which will be used to retrieve LoRA models for the generation process.

---

### Guidelines:

1. **Keywords Definition:**
   - A *keyword* can be a single word or a multi-word phrase (e.g., "Golden Retriever" is valid, but splitting it into "Golden" and "Retriever" is incorrect).
   - Keywords should capture distinct and meaningful concepts from the input prompt, avoiding overly generic terms like "and," "of," "the."

2. **Keyword Purpose:**
   - These keywords will be used to search for relevant LoRA models, so they should be specific and directly related to the objects, subjects, styles, or themes mentioned in the input prompt.

3. **Maximum Keywords:**
   - Extract up to **{max_keywords}** keywords from the prompt. If fewer meaningful keywords are present, return only those.

4. **Input Format:**
   - You will be given a *user_prompt* (the text for image generation).

5. **Output Format:**
   - Provide a numbered list of keywords. Each keyword should be on its own line.

---

### Example:

**User Prompt:**  
"A majestic lion standing on a cliff at sunset, painted in the style of Van Gogh."

**Output Keywords:**  
1. Majestic lion  
2. Cliff  
3. Sunset  
4. Van Gogh style  

---

**User Prompt:**  
"{user_prompt}"  

**Output Keywords:**  
(Provide your response here based on the given user prompt, following the same format.)
    """

    # Send the request to the Ollama API
    try:
        response: ChatResponse = chat(model="phi3", messages=[
            {"role": "user", "content": llama_prompt}
        ])

        # Extract the generated text
        generated_text = response.message.content
        # print (generated_text)

        # Extract keywords from the generated text
        keywords = []
        for line in generated_text.splitlines():
            # print(line)
            if line.strip().startswith("-"): break
            if line.split(".")[0].isdigit():
                # Remove the number prefix and strip whitespace
                keyword = line.split('.', 1)[-1].strip()
                # print(keyword)
                if keyword:
                    keywords.append(keyword)
                    if len(keywords) >= max_keywords:
                        break

        return keywords

    except Exception as e:
        print("Error while connecting to the Ollama API:", str(e))
        return []

# Example usage
if __name__ == "__main__":
    user_prompt = "A futuristic cityscape with flying cars, glowing neon signs, and towering skyscrapers."
    max_keywords = 5
    keywords = parse_prompt(user_prompt, max_keywords)
    print("Extracted Keywords:", keywords)
