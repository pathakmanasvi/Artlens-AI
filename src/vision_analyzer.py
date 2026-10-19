import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

# 1. Load environment variables FIRST
load_dotenv()

# 2. Initialize the client using the loaded API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 3. Point to your image file
sample_image = "monalisa.jpg"


def analyze_artwork(image_path: str):
  """Loads an image and uses the modern Gemini API to identify and describe it."""
  if not os.path.exists(image_path):
    print(f"Error: Image path '{image_path}' does not exist.")
    return

  print(f"Loading image from {image_path}...")
  img = Image.open(image_path)

  # Craft a structured prompt aligned with art analysis goals
  prompt = (
      "Analyze this image as an art historian and visual expert. Provide:"
      "\n1. **Identification:** Title, possible artist or era, and estimated"
      " creation date/century."
      "\n2. **Visual & Formal Description:** Breakdown of composition, lighting,"
      " brushwork, palette, and texture."
      "\n3. **Interpretive Context:** Potential themes, symbolic elements, and"
      " cultural resonance."
  )

  print("Sending image to Gemini Multimodal API for analysis...\n")

  response = client.models.generate_content(
      model="gemini-3.8-flash",  # Updated model version
      contents=[img, prompt],
  )

  print("--- ArtLens AI Analysis Result ---")
  print(response.text)
  print("-----------------------------------")


if __name__ == "__main__":
  if os.path.exists(sample_image):
    analyze_artwork(sample_image)
  else:
    print(
        f"Sample image not found at '{sample_image}'. Please check your image"
        " path."
    )