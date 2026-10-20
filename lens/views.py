import os
from django.shortcuts import render
from google import genai
from PIL import Image

# Initialize the Gemini client using the environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_view(request):
    analysis_result = None
    if request.method == "POST" and request.FILES.get("artwork_image"):
        image_file = request.FILES["artwork_image"]
        
        # Open image using Pillow
        img = Image.open(image_file)
        
        prompt = (
            "Analyze this image as an art historian and visual expert. Provide:\n"
            "1. **Identification:** Title, possible artist or era, and estimated creation date/century.\n"
            "2. **Visual & Formal Description:** Breakdown of composition, lighting, brushwork, palette, and texture.\n"
            "3. **Interpretive Context:** Potential themes, symbolic elements, and cultural resonance."
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[img, prompt],
        )
        analysis_result = response.text

    return render(request, "lens/analyze.html", {"analysis_result": analysis_result})