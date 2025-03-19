import streamlit as st
import json
import os
from dotenv import load_dotenv, set_key

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="Therapy AI - Firebase Setup", page_icon="🔧")
    
    st.title("Firebase Configuration Setup")
    st.write("Configure your Firebase credentials for the Therapy AI application.")
    
    # Check if credentials file exists
    credentials_file = "firebase_credentials.json"
    env_file = ".env"
    
    if os.path.exists(credentials_file):
        st.success(f"Firebase credentials file exists at {credentials_file}")
        
        # Display current configuration (hiding sensitive info)
        with open(credentials_file, "r") as f:
            creds = json.load(f)
            
        st.subheader("Current Firebase Project Details")
        st.write(f"Project ID: {creds.get('project_id', 'Not found')}")
        st.write(f"Client Email: {creds.get('client_email', 'Not found')}")
        
        # Option to update credentials
        st.subheader("Update Credentials")
    else:
        st.warning(f"Firebase credentials file not found. Let's create one.")
        st.subheader("Enter Firebase Credentials")
    
    # Firebase credentials form
    with st.form("firebase_credentials"):
        # Project configuration
        project_id = st.text_input("Project ID", 
                                  help="Your Firebase project ID")
        
        # Service account details
        st.subheader("Service Account Details")
        st.write("You can get these values from your Firebase service account JSON file.")
        
        type_input = st.text_input("Type", value="service_account",
                                 help="Usually 'service_account'")
        project_id_input = st.text_input("Project ID (from service account)",
                                       help="Project ID from your service account")
        private_key_id = st.text_input("Private Key ID",
                                      help="Private key ID from your service account")
        private_key = st.text_area("Private Key", 
                                  help="Your private key (starts with '-----BEGIN PRIVATE KEY-----')")
        client_email = st.text_input("Client Email",
                                    help="Service account email")
        client_id = st.text_input("Client ID",
                                 help="Client ID from your service account")
        auth_uri = st.text_input("Auth URI", 
                               value="https://accounts.google.com/o/oauth2/auth",
                               help="Usually 'https://accounts.google.com/o/oauth2/auth'")
        token_uri = st.text_input("Token URI",
                                value="https://oauth2.googleapis.com/token",
                                help="Usually 'https://oauth2.googleapis.com/token'")
        auth_provider_x509_cert_url = st.text_input("Auth Provider X509 Cert URL",
                                                  value="https://www.googleapis.com/oauth2/v1/certs",
                                                  help="Usually 'https://www.googleapis.com/oauth2/v1/certs'")
        client_x509_cert_url = st.text_input("Client X509 Cert URL",
                                           help="Client certificate URL from your service account")
        
        # Firebase storage bucket
        st.subheader("Firebase Storage Configuration")
        storage_bucket = st.text_input("Storage Bucket", 
                                      help="Your Firebase storage bucket name (usually 'project-id.appspot.com')")
        
        # Submit button
        submitted = st.form_submit_button("Save Configuration")
        
        if submitted:
            try:
                # Create credentials JSON
                credentials = {
                    "type": type_input,
                    "project_id": project_id_input,
                    "private_key_id": private_key_id,
                    "private_key": private_key.replace("\\n", "\n") if private_key else "",
                    "client_email": client_email,
                    "client_id": client_id,
                    "auth_uri": auth_uri,
                    "token_uri": token_uri,
                    "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
                    "client_x509_cert_url": client_x509_cert_url
                }
                
                # Save credentials to file
                with open(credentials_file, "w") as f:
                    json.dump(credentials, f, indent=2)
                
                # Update environment variables
                env_vars = {
                    "FIREBASE_PROJECT_ID": project_id,
                    "FIREBASE_STORAGE_BUCKET": storage_bucket
                }
                
                # Create or update .env file
                with open(env_file, "a+") as f:
                    f.seek(0)
                    content = f.read()
                    for key, value in env_vars.items():
                        if f"{key}=" in content:
                            set_key(env_file, key, value)
                        else:
                            f.write(f"{key}={value}\n")
                
                st.success("Firebase configuration saved successfully!")
                st.info("You can now run the main application with Firebase integration.")
                
            except Exception as e:
                st.error(f"Error saving configuration: {str(e)}")
    
    # Instructions for getting Firebase credentials
    with st.expander("How to get Firebase credentials"):
        st.write("""
        1. Go to the [Firebase Console](https://console.firebase.google.com/)
        2. Create a new project or select an existing one
        3. Go to Project Settings > Service accounts
        4. Click "Generate new private key" to download a JSON file
        5. Use the values from that JSON file to fill in this form
        6. For the Storage Bucket, go to the Storage section in Firebase Console and note the bucket name
        """)
        
    # Instructions for enabling Firebase services
    with st.expander("Required Firebase Services"):
        st.write("""
        Make sure you have enabled the following Firebase services:
        
        1. **Firestore Database**:
           - Go to Firestore Database in the Firebase Console
           - Create a database in test mode
        
        2. **Storage**:
           - Go to Storage in the Firebase Console
           - Set up storage in test mode
        
        3. **Authentication** (optional):
           - Set up authentication if you plan to have user accounts
        """)

if __name__ == "__main__":
    main() 