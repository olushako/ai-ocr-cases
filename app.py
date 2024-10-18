import streamlit as st
import args
import ai
import boto3
import time 
import prompts
from PIL import Image
import pytesseract
import prices 

@st.cache_resource
def initialize_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.aws_region
    )

@st.cache_resource
def initialize_rekognition_client():
    return boto3.client(
        service_name='rekognition',
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.aws_region
    )

def process_images(bedrock_client, text, photo_content, model_id, input_price, output_price):
    start = time.time()
    response, input_tokens, output_tokens = ai.query_bedrock_model_with_images(bedrock_client,text,photo_content,model_id)
    total_tokens = input_tokens+output_tokens
    cost = (input_tokens/1000)*input_price+(output_tokens/1000)*output_price
    duration = time.time()-start
    return response, cost, duration, total_tokens

def detect_text_in_image(image_bytes):
    # Create an AWS Rekognition client
    rekognition_client = initialize_rekognition_client()
    # Call the detect_text method
    response = rekognition_client.detect_text(
        Image={'Bytes': image_bytes}
    )
    detected_texts = []
    for text in response['TextDetections']:
        detected_texts.append({
            text['DetectedText']
        })
    return str(detected_texts)


def main():
    
    st.write('---')
    st.header("OCR & Data Extraction :blue[демонстрація] :sunglasses:")
    st.write('---')

    st.header("Класичне розпізнавання", divider="gray")
    uploaded_files1 = st.file_uploader("Select image", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload1")
    submit_button1 = st.button("Submit",key="sb1")
    if submit_button1:
        bedrock_client = initialize_bedrock_client()
        image_bytes = uploaded_files1[0].read()
        image = Image.open(uploaded_files1[0])
        detected_texts = detect_text_in_image(image_bytes)
        st.write(detected_texts)
    
    st.write("---")
    st.header("AI розпізнавання", divider="gray")
    uploaded_files2 = st.file_uploader("Select image", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload2")
    model1 = st.selectbox("Виберіть АІ модель", 
                         ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", 
                          "anthropic.claude-3-5-sonnet-20240620-v1:0","anthropic.claude-3-opus-20240229-v1:0"),
                          key='model1')
    
    prompt1 = st.text_area("Команда для AI", prompts.basic_recognition_prompt, key='prompt1')
    
    submit_button2 = st.button("Submit",key="sb2")
    if submit_button2:
        if uploaded_files2:
            bedrock_client = initialize_bedrock_client()
            result, input, output = ai.query_bedrock_with_multiple_images(bedrock_client,prompt1, uploaded_files2, model1)
        
        image = Image.open(uploaded_files2[0])
        st.image(image, caption='Картинка', use_column_width=True)
        st.write(result)
        st.write(input)
        st.write(output)
        st.write((input*prices.model_costs[model1]['input']/1000+output*prices.model_costs[model1]['output']/1000)*args.exchange_rate)

    else:
        bedrock_client = initialize_bedrock_client()


    st.write("---")
    st.header("Розпізнавання рукописного тексту ", divider="gray")
    uploaded_files3 = st.file_uploader("Select image", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload3")
    model2 = st.selectbox("Виберіть АІ модель", 
                         ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", 
                          "anthropic.claude-3-5-sonnet-20240620-v1:0","anthropic.claude-3-opus-20240229-v1:0"),
                          key='model2')
    
    prompt2 = st.text_area("Команда для AI", prompts.handwritten_recognition_prompt, key = 'prompt2')
    
    submit_button3 = st.button("Submit",key="sb3")
    if submit_button3:
        if uploaded_files3:
            bedrock_client = initialize_bedrock_client()
            result, input, output = ai.query_bedrock_with_multiple_images(bedrock_client,prompt2, uploaded_files3, model2)
        
        image = Image.open(uploaded_files3[0])
        st.image(image, caption='Картинка', use_column_width=True)
        st.write(result)
        st.write(input)
        st.write(output)
        st.write((input*prices.model_costs[model1]['input']/1000+output*prices.model_costs[model2]['output']/1000)*args.exchange_rate)

    else:
        bedrock_client = initialize_bedrock_client()

    st.write("---")
    st.header("Класифікація документів ", divider="gray")
    uploaded_files4 = st.file_uploader("Select image", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload4")
    model3 = st.selectbox("Виберіть АІ модель", 
                         ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", 
                          "anthropic.claude-3-5-sonnet-20240620-v1:0","anthropic.claude-3-opus-20240229-v1:0"),
                          key='model3')
    
    prompt3 = st.text_area("Команда для AI", prompts.data_classification_prompt, key = 'prompt3')
    
    submit_button4 = st.button("Submit",key="sb4")
    if submit_button4:
        if uploaded_files4:
            bedrock_client = initialize_bedrock_client()
            result, input, output = ai.query_bedrock_with_multiple_images(bedrock_client,prompt3, uploaded_files4, model3)
        
        image = Image.open(uploaded_files4[0])
        st.image(image, caption='Картинка', use_column_width=True)
        st.write(result)
        st.write(input)
        st.write(output)
        st.write((input*prices.model_costs[model3]['input']/1000+output*prices.model_costs[model3]['output']/1000)*args.exchange_rate)

    else:
        bedrock_client = initialize_bedrock_client()


    st.write("---")
    st.header("Вивантаження даних ", divider="gray")
    uploaded_files5 = st.file_uploader("Select image", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload5")
    model4 = st.selectbox("Виберіть АІ модель", 
                         ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", 
                          "anthropic.claude-3-5-sonnet-20240620-v1:0","anthropic.claude-3-opus-20240229-v1:0"),
                          key='model4')
    
    prompt4 = st.text_area("Команда для AI", prompts.data_extraction_prompt, key = 'prompt4')
    
    submit_button5 = st.button("Submit",key="sb5")
    if submit_button5:
        if uploaded_files5:
            bedrock_client = initialize_bedrock_client()
            result, input, output = ai.query_bedrock_with_multiple_images(bedrock_client,prompt4, uploaded_files5, model4)
        
        image = Image.open(uploaded_files5[0])
        st.image(image, caption='Картинка', use_column_width=True)
        st.write(result)
        st.write(input)
        st.write(output)
        st.write((input*prices.model_costs[model4]['input']/1000+output*prices.model_costs[model4]['output']/1000)*args.exchange_rate)

    else:
        bedrock_client = initialize_bedrock_client()

    st.write("---")
    st.header("Перевірки ", divider="gray")
    uploaded_files6 = st.file_uploader("Select image", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key="upload6")
    model5 = st.selectbox("Виберіть АІ модель", 
                         ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", 
                          "anthropic.claude-3-5-sonnet-20240620-v1:0","anthropic.claude-3-opus-20240229-v1:0"),
                          key='model5')
    
    prompt5 = st.text_area("Команда для AI", prompts.data_validation_prompt, key = 'prompt5')
    
    submit_button6 = st.button("Submit",key="sb6")
    if submit_button6:
        if uploaded_files6:
            bedrock_client = initialize_bedrock_client()
            result, input, output = ai.query_bedrock_with_multiple_images(bedrock_client,prompt5, uploaded_files6, model5)
        
        image = Image.open(uploaded_files6[0])
        st.image(image, caption='Картинка', use_column_width=True)
        st.write(result)
        st.write(input)
        st.write(output)
        st.write((input*prices.model_costs[model5]['input']/1000+output*prices.model_costs[model5]['output']/1000)*args.exchange_rate)

    else:
        bedrock_client = initialize_bedrock_client()


if __name__ == '__main__':
    main()
