import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import subprocess
import json
import requests
import time
from PyPDF2 import PdfReader, PdfWriter

# Page config
st.set_page_config(
    page_title="PDF Gemini Converter",
    page_icon="📄",
    layout="wide"
)

class GeminiConverter:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"
        
    def upload_file_to_gemini(self, file_path):
        """Upload file using REST API"""
        print(f"Starting upload for: {file_path}")
        try:
            return self.upload_file_multipart(file_path)
        except Exception as e:
            print(f"Multipart upload failed: {e}, trying resumable upload...")
            return self.upload_file_resumable(file_path)
    
    def upload_file_multipart(self, file_path):
        """Simple multipart upload"""
        url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"
        
        mime_type = self.get_mime_type(file_path)
        file_name = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            files = {
                'metadata': (None, json.dumps({
                    'file': {'display_name': file_name}
                }), 'application/json'),
                'data': (file_name, f, mime_type)
            }
            
            response = requests.post(url, files=files)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Multipart upload failed: {response.status_code} - {response.text}")
        
        file_info = response.json()
        if 'file' in file_info and 'uri' in file_info['file']:
            return file_info['file']
        elif 'uri' in file_info:
            return file_info
        else:
            raise Exception(f"No URI in multipart response: {file_info}")
    
    def upload_file_resumable(self, file_path):
        """Resumable upload as fallback"""
        url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"
        
        file_size = os.path.getsize(file_path)
        mime_type = self.get_mime_type(file_path)
        
        headers = {
            'X-Goog-Upload-Protocol': 'resumable',
            'X-Goog-Upload-Command': 'start',
            'X-Goog-Upload-Header-Content-Length': str(file_size),
            'X-Goog-Upload-Header-Content-Type': mime_type,
            'Content-Type': 'application/json'
        }
        
        metadata = {
            'file': {
                'display_name': os.path.basename(file_path)
            }
        }
        
        response = requests.post(url, headers=headers, json=metadata)
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to initiate resumable upload: {response.status_code} - {response.text}")
        
        upload_url = response.headers.get('X-Goog-Upload-URL')
        if not upload_url:
            raise Exception("No upload URL received from initiation")
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        upload_headers = {
            'Content-Length': str(file_size),
            'X-Goog-Upload-Offset': '0',
            'X-Goog-Upload-Command': 'upload, finalize'
        }
        
        response = requests.post(upload_url, headers=upload_headers, data=file_data)
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to upload file content: {response.status_code} - {response.text}")
        
        try:
            file_info = response.json()
            if 'file' in file_info and 'uri' in file_info['file']:
                return file_info['file']
            elif 'uri' in file_info:
                return file_info
            else:
                raise Exception(f"No URI in resumable response: {file_info}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response from upload: {response.text}")
    
    def get_mime_type(self, file_path):
        """Get MIME type based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    def wait_for_file_processing(self, file_uri):
        """Wait for file to be processed"""
        if file_uri.startswith('files/'):
            file_name = file_uri.split('/')[-1]
        elif '/' in file_uri:
            file_name = file_uri.split('/')[-1]
        else:
            file_name = file_uri
            
        url = f"https://generativelanguage.googleapis.com/v1beta/files/{file_name}?key={self.api_key}"
        
        max_wait_time = 300
        wait_time = 0
        
        while wait_time < max_wait_time:
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    if wait_time == 0:
                        raise Exception(f"Failed to check file status: {response.status_code} - {response.text}")
                    else:
                        time.sleep(2)
                        wait_time += 2
                        continue
                
                file_info = response.json()
                state = file_info.get('state', 'UNKNOWN')
                
                if state == 'ACTIVE':
                    return file_info
                elif state == 'FAILED':
                    raise Exception(f"File processing failed: {file_info}")
                elif state in ['PROCESSING', 'UNKNOWN']:
                    time.sleep(2)
                    wait_time += 2
                else:
                    raise Exception(f"Unknown file state: {state}")
                    
            except requests.RequestException as e:
                if wait_time == 0:
                    raise Exception(f"Network error checking file status: {str(e)}")
                else:
                    time.sleep(2)
                    wait_time += 2
        
        raise Exception(f"File processing timeout after {max_wait_time} seconds")

    def generate_content_with_api(self, prompt, file_uri):
        """Generate content using REST API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        file_data_part = {}
        if file_uri.startswith('files/'):
            file_data_part = {"file_data": {"file_uri": file_uri}}
        else:
            if not file_uri.startswith('files/'):
                file_uri = f"files/{file_uri}" if '/' not in file_uri else file_uri
            file_data_part = {"file_data": {"file_uri": file_uri}}
        
        request_body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        file_data_part
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 32768,
                "responseMimeType": "text/plain"
            }
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, json=request_body, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        
        return response.json()

    def convert_file(self, file_path, prompt):
        """Convert file to text"""
        try:
            # Verify file exists and has content
            if not os.path.exists(file_path):
                raise Exception(f"File does not exist: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise Exception(f"File is empty: {file_path}")
            
            print(f"Converting file: {file_path} (Size: {file_size} bytes)")
            
            # Upload file
            file_info = self.upload_file_to_gemini(file_path)
            
            # Get file URI
            file_uri = None
            if isinstance(file_info, dict):
                file_uri = file_info.get('uri') or file_info.get('name')
            
            if not file_uri:
                raise Exception(f"No file URI found in response: {file_info}")
            
            print(f"File uploaded with URI: {file_uri}")
            
            # Wait for file processing
            processed_file = self.wait_for_file_processing(file_uri)
            if not processed_file:
                raise Exception("File processing failed")
            
            print("File processing completed, generating content...")
            
            # Generate content
            response = self.generate_content_with_api(prompt, file_uri)
            
            # Extract text from response
            if 'candidates' in response and response['candidates']:
                candidate = response['candidates'][0]
                
                finish_reason = candidate.get('finishReason', '')
                if 'SAFETY' in finish_reason:
                    raise Exception("Content blocked by safety filters")
                
                if 'content' in candidate and 'parts' in candidate['content']:
                    text_parts = []
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            text_parts.append(part['text'])
                    if text_parts:
                        result_text = ''.join(text_parts)
                        print(f"Successfully extracted {len(result_text)} characters")
                        return result_text
            
            raise Exception("No valid text content in response")
                        
        except Exception as e:
            print(f"Conversion error: {str(e)}")
            raise Exception(f"Conversion failed: {str(e)}")

def load_api_key_from_storage():
    """Load API key from localStorage using JavaScript"""
    html_code = """
    <script>
    function getApiKey() {
        const apiKey = localStorage.getItem('gemini_api_key');
        return apiKey || '';
    }
    
    function sendApiKey() {
        const apiKey = getApiKey();
        window.parent.postMessage({
            type: 'API_KEY_LOADED',
            data: apiKey
        }, '*');
    }
    
    // Send API key when component loads
    sendApiKey();
    </script>
    <div id="api-loader" style="display: none;">Loading API key...</div>
    """
    
    result = components.html(html_code, height=0)
    return result

def save_api_key_to_storage(api_key):
    """Save API key to localStorage using JavaScript"""
    html_code = f"""
    <script>
    function saveApiKey() {{
        localStorage.setItem('gemini_api_key', '{api_key}');
        window.parent.postMessage({{
            type: 'API_KEY_SAVED',
            data: 'success'
        }}, '*');
    }}
    
    // Save API key when component loads
    saveApiKey();
    </script>
    <div id="api-saver" style="display: none;">Saving API key...</div>
    """
    
    components.html(html_code, height=0)

def check_saved_api_key():
    """Check if there's a saved API key in localStorage"""
    html_code = """
    <script>
    function checkSavedApiKey() {
        setTimeout(() => {
            const savedKey = localStorage.getItem('gemini_api_key');
            if (savedKey && savedKey.trim()) {
                // Send the API key to Streamlit
                window.parent.postMessage({
                    type: 'LOAD_SAVED_API_KEY',
                    data: savedKey
                }, '*');
            }
        }, 100);
    }
    
    checkSavedApiKey();
    </script>
    <div style="display: none;">Checking for saved API key...</div>
    """
    
    return components.html(html_code, height=0)
    """Split PDF into chunks"""
    # Create temp file for original PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        pdf = PdfReader(temp_path)
        total_pages = len(pdf.pages)
        
        if total_pages <= chunk_size:
            # Return original file if small enough
            return [temp_path], total_pages
        
        # Split into chunks
        chunk_files = []
        num_chunks = (total_pages + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_page = i * chunk_size
            end_page = min((i + 1) * chunk_size, total_pages)

            output = PdfWriter()
            for page in range(start_page, end_page):
                output.add_page(pdf.pages[page])

            # Create chunk file with better handling
            try:
                chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_part{i+1}.pdf', prefix='pdf_chunk_')
                chunk_path = chunk_file.name
                
                # Write PDF content
                output.write(chunk_file)
                chunk_file.close()
                
                # Verify file was created and has content
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                    chunk_files.append(chunk_path)
                    print(f"Created chunk {i+1}: {chunk_path} ({os.path.getsize(chunk_path)} bytes)")
                else:
                    print(f"Failed to create chunk {i+1}: {chunk_path}")
                    
            except Exception as chunk_error:
                print(f"Error creating chunk {i+1}: {chunk_error}")
                continue
        
        # Remove original temp file only if we created chunks
        if len(chunk_files) > 0 and os.path.exists(temp_path):
            os.remove(temp_path)
            
        return chunk_files, total_pages
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"PDF splitting failed: {str(e)}")

def save_image_file(uploaded_file):
    """Save uploaded image to temp file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def convert_to_word(text_content):
    """Convert text to Word document using pandoc"""
    try:
        result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Pandoc not found")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding='utf-8') as md_file:
            md_file.write(text_content)
            md_path = md_file.name
        
        docx_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
        docx_path = docx_file.name
        docx_file.close()
        
        subprocess.run(
            ["pandoc", md_path, "-o", docx_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        with open(docx_path, 'rb') as f:
            docx_content = f.read()
        
        os.unlink(md_path)
        os.unlink(docx_path)
        
        return docx_content
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"Pandoc conversion failed: {e.stderr}")
    except Exception as e:
        raise Exception(f"Word conversion failed: {str(e)}")

def split_pdf(uploaded_file, chunk_size=8):
    """Split PDF into chunks"""
    # Create temp file for original PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        pdf = PdfReader(temp_path)
        total_pages = len(pdf.pages)
        
        if total_pages <= chunk_size:
            # Return original file if small enough
            return [temp_path], total_pages
        
        # Split into chunks
        chunk_files = []
        num_chunks = (total_pages + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_page = i * chunk_size
            end_page = min((i + 1) * chunk_size, total_pages)

            output = PdfWriter()
            for page in range(start_page, end_page):
                output.add_page(pdf.pages[page])

            # Create chunk file with better handling
            try:
                chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_part{i+1}.pdf', prefix='pdf_chunk_')
                chunk_path = chunk_file.name
                
                # Write PDF content
                output.write(chunk_file)
                chunk_file.close()
                
                # Verify file was created and has content
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                    chunk_files.append(chunk_path)
                    print(f"Created chunk {i+1}: {chunk_path} ({os.path.getsize(chunk_path)} bytes)")
                else:
                    print(f"Failed to create chunk {i+1}: {chunk_path}")
                    
            except Exception as chunk_error:
                print(f"Error creating chunk {i+1}: {chunk_error}")
                continue
        
        # Remove original temp file only if we created chunks
        if len(chunk_files) > 0 and os.path.exists(temp_path):
            os.remove(temp_path)
            
        return chunk_files, total_pages
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"PDF splitting failed: {str(e)}")

def save_image_file(uploaded_file):
    """Save uploaded image to temp file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def convert_to_word(text_content):
    """Convert text to Word document using pandoc"""
    try:
        result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Pandoc not found")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding='utf-8') as md_file:
            md_file.write(text_content)
            md_path = md_file.name
        
        docx_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
        docx_path = docx_file.name
        docx_file.close()
        
        subprocess.run(
            ["pandoc", md_path, "-o", docx_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        with open(docx_path, 'rb') as f:
            docx_content = f.read()
        
        os.unlink(md_path)
        os.unlink(docx_path)
        
        return docx_content
        
    except subprocess.CalledProcessError as e:
        raise Exception(f"Pandoc conversion failed: {e.stderr}")
    except Exception as e:
        raise Exception(f"Word conversion failed: {str(e)}")

def main():
    st.title("📄 PDF Gemini Converter")
    st.markdown("Convert PDF files and images to text using Google's Gemini AI")
    st.markdown("---")
    
    with st.sidebar:
        st.header("🔑 Configuration")
        api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
        
        if api_key:
            st.success("✅ API Key provided")
        else:
            st.warning("⚠️ Please enter your API key")
            st.markdown("Get your API key from [Google AI Studio](https://aistudio.google.com/)")
    
    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar to continue.")
        st.info("📋 **How to get an API key:**\n1. Visit [Google AI Studio](https://aistudio.google.com/)\n2. Create a new project or select existing one\n3. Generate an API key\n4. Enter the API key in the sidebar")
        st.stop()
    
    st.header("📁 Upload File")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file",
        type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="Upload a PDF or image file to convert to text"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        file_type = uploaded_file.type
        file_size = len(uploaded_file.getvalue())
        st.info(f"📊 File type: {file_type} | Size: {file_size/1024:.1f} KB")
        
        if st.button("🔄 Convert to Text", type="primary"):
            try:
                converter = GeminiConverter(api_key)
                
                prompt = """
                Hãy nhận diện và gõ lại [CHÍNH XÁC] nội dung trong file thành văn bản, tất cả công thức Toán được bọc trong dấu $
                [TUYỆT ĐỐI] không thêm nội dung khác ngoài nội dung trong file, [CHỈ ĐƯỢC PHÉP] gõ lại nội dung thành văn bản.
                """
                
                with st.spinner("🔄 Converting file to text..."):
                    if file_type == "application/pdf":
                        try:
                            chunk_files, total_pages = split_pdf(uploaded_file)
                            st.info(f"📄 PDF has {total_pages} pages, split into {len(chunk_files)} parts")
                            
                            # Debug info
                            st.write("📋 **Debug Info:**")
                            for i, chunk_file in enumerate(chunk_files):
                                if os.path.exists(chunk_file):
                                    size = os.path.getsize(chunk_file)
                                    st.write(f"  • Part {i+1}: {os.path.basename(chunk_file)} ({size/1024:.1f} KB)")
                                else:
                                    st.write(f"  • Part {i+1}: ❌ File not found")
                            
                            all_text = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, chunk_file in enumerate(chunk_files):
                                try:
                                    status_text.text(f"Processing part {i+1}/{len(chunk_files)}...")
                                    
                                    # Verify file exists before processing
                                    if not os.path.exists(chunk_file):
                                        st.error(f"❌ Chunk file {i+1} not found: {chunk_file}")
                                        continue
                                    
                                    file_size = os.path.getsize(chunk_file)
                                    if file_size == 0:
                                        st.error(f"❌ Chunk file {i+1} is empty")
                                        continue
                                    
                                    with st.expander(f"📄 Processing Part {i+1}", expanded=False):
                                        st.write(f"File path: `{chunk_file}`")
                                        st.write(f"File size: {file_size/1024:.1f} KB")
                                    
                                    text = converter.convert_file(chunk_file, prompt)
                                    if text and text.strip():
                                        all_text.append(text)
                                        st.success(f"✅ Part {i+1} processed successfully ({len(text)} characters)")
                                    else:
                                        st.warning(f"⚠️ Part {i+1} returned empty text")
                                    
                                    progress_bar.progress((i + 1) / len(chunk_files))
                                    
                                except Exception as e:
                                    st.error(f"❌ Error processing part {i+1}: {str(e)}")
                                    with st.expander("🔍 Error Details", expanded=False):
                                        st.write(f"File path: `{chunk_file}`")
                                        st.write(f"File exists: {os.path.exists(chunk_file)}")
                                        if os.path.exists(chunk_file):
                                            st.write(f"File size: {os.path.getsize(chunk_file)} bytes")
                                        st.write(f"Error: {str(e)}")
                                finally:
                                    # Clean up chunk file
                                    try:
                                        if os.path.exists(chunk_file):
                                            os.remove(chunk_file)
                                    except Exception as cleanup_error:
                                        st.warning(f"⚠️ Could not clean up file {chunk_file}: {cleanup_error}")
                            
                            if all_text:
                                final_text = "\n\n--- Page Break ---\n\n".join(all_text)
                                st.session_state.converted_text = final_text
                                status_text.text("🎉 Conversion completed!")
                            else:
                                st.error("❌ No text could be extracted from the PDF")
                                st.info("💡 **Troubleshooting Tips:**\n- Try with a smaller PDF\n- Check if PDF contains text (not just images)\n- Verify your API key and quota\n- Some PDFs may be password protected or corrupted")
                                st.stop()
                                
                        except Exception as pdf_error:
                            st.error(f"❌ PDF processing failed: {str(pdf_error)}")
                            with st.expander("🔍 Error Details", expanded=True):
                                st.write(f"Error type: {type(pdf_error).__name__}")
                                st.write(f"Error message: {str(pdf_error)}")
                            st.info("💡 **Try:**\n- Using a different PDF file\n- Converting to images first\n- Checking file integrity\n- Using a smaller file")
                            st.stop()
                    
                    else:
                        image_file = save_image_file(uploaded_file)
                        try:
                            text = converter.convert_file(image_file, prompt)
                            st.session_state.converted_text = text
                        finally:
                            if os.path.exists(image_file):
                                os.remove(image_file)
                
                st.success("🎉 Conversion completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Conversion failed: {str(e)}")
                st.info("💡 Tips:\n- Check your API key is valid\n- Ensure you have quota remaining\n- Some content may be blocked by safety filters")
    
    if 'converted_text' in st.session_state and st.session_state.converted_text:
        st.header("📋 Results")
        
        st.text_area(
            "Converted Text",
            value=st.session_state.converted_text,
            height=400,
            help="The converted text from your file"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy Text"):
                st.code(st.session_state.converted_text, language=None)
                st.info("💡 Select all text above and copy manually (Ctrl+A, Ctrl+C)")
        
        with col2:
            st.download_button(
                label="📄 Download Text File",
                data=st.session_state.converted_text,
                file_name="converted_text.txt",
                mime="text/plain"
            )
        
        with col3:
            try:
                result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    if st.button("💾 Generate Word Document"):
                        try:
                            with st.spinner("Creating Word document..."):
                                docx_content = convert_to_word(st.session_state.converted_text)
                                
                                st.download_button(
                                    label="📄 Download Word Document",
                                    data=docx_content,
                                    file_name="converted_text.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Word conversion failed: {str(e)}")
                            st.info("💡 You can copy the text and paste it into a Word document manually")
                else:
                    st.warning("⚠️ Pandoc not available. Word conversion disabled.")
            except:
                st.warning("⚠️ Pandoc not available. Word conversion disabled.")

if __name__ == "__main__":
    main()
