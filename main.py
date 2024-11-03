import base64
import openai
import streamlit as st
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
from langchain.chains import TransformChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.callbacks import get_openai_callback

@dataclass
class VisionPassportExtractionPrompt:
    template: str = """
        You are an expert at information extraction from images of passports.
        Extract the following details from the image of a passport, and respond strictly in JSON format:
        {
            "passport_number": "actual passport number or N/A",
            "issuing_country": "country or N/A",
            "name": "name or N/A",
            "date_of_birth": "YYYY-MM-DD or N/A",
            "nationality": "nationality or N/A",
            "gender": "gender or N/A",
            "expiration_date": "N/A or YYYY-MM-DD",
            "place_of_birth": "place of birth or N/A"
        }
        Do not include any additional text or explanations outside this JSON structure.
    """

class PassportInformation(BaseModel):
    passport_number: str
    issuing_country: str
    name: str
    date_of_birth: str
    nationality: str
    gender: str
    expiration_date: str
    place_of_birth: str

class VisionPassportExtractionChain:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self.set_up_chain()

    @staticmethod
    def load_image(path: dict) -> dict:
        """Load image and encode it as base64."""
        def encode_image(path):
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        image_base64 = encode_image(path["image_path"])
        return {"image": image_base64}

    def set_up_chain(self):
        prompt = VisionPassportExtractionPrompt()
        parser = JsonOutputParser(pydantic_object=PassportInformation)
        load_image_chain = TransformChain(
            input_variables=["image_path"],
            output_variables=["image"],
            transform=self.load_image,
        )

    def passport_model_chain(inputs: dict) -> dict:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt.template},
                {"role": "user", "content": inputs['image']}
            ]
        )
        try:
            parsed_response = parser.parse(response['choices'][0]['message']['content'])
            return parsed_response
        except (ValidationError, OutputParserException) as e:
            st.write("Raw Response for Debugging:", response['choices'][0]['message']['content'])
            return {
                "error": "Parsing failed. Please ensure the format follows the required structure.",
                "details": str(e)
            }
        return load_image_chain | passport_model_chain

    def run_and_count_tokens(self, input_dict: dict):
        with get_openai_callback() as cb:
            result = self.chain.invoke(input_dict)
        return result, cb

def main():
    st.title("Passport Information Extraction Tool")
    st.write("Upload an image of a passport to extract information.")

    uploaded_image = st.file_uploader("Choose a passport image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with open("temp_passport_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(img, caption="Uploaded Image", use_column_width=True)

        openai.api_key = "OPENAI_KEY"
        llm = openai.ChatCompletion

        extraction_chain = VisionPassportExtractionChain(llm)
        input_data = {"image_path": "temp_passport_image.jpg"}
        result, cb = extraction_chain.run_and_count_tokens(input_data)

        st.subheader("Extracted Passport Information")
        st.json(result)
        st.write("Token usage:", cb.total_tokens)

if __name__ == "__main__":
    main()
