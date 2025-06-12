import asyncio
import audioop
import base64
import logging
import os
import socket
import uuid

from aiohttp import web
from aiosip import AioSIP
from aiosip.dialog import Dialog
from aiosip.rtp import Rtp
from dotenv import load_dotenv

# Google Cloud Clients
from google.cloud import speech
from google.cloud import texttospeech
import google.generativeai as genai

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

# GCE VM static IP. CRITICAL for SIP and Telnyx Webhooks.
VM_STATIC_IP = os.getenv("VM_STATIC_IP")
if not VM_STATIC_IP:
    raise ValueError("VM_STATIC_IP environment variable not set.")

SIP_PORT = 5060
WEBHOOK_PORT = 8080

# Google Cloud Credentials should be handled by the environment variable
# GOOGLE_APPLICATION_CREDENTIALS pointing to service-account.json
# or by the GCE metadata server's IAM role.

# Initialize Google Cloud Clients
try:
    speech_client = speech.SpeechClient()
    tts_client = texttospeech.TextToSpeechClient()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest') 
    logging.info("Google Cloud clients initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud clients: {e}")
    exit(1)


# --- Core Logic: A class to manage a single call ---

class VoiceAgentDialog(Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rtp_server = None
        self.call_ongoing = False
        self.audio_queue = asyncio.Queue()
        self.conversation_history = []
        logging.info(f"New VoiceAgentDialog created for call {self.dialog_id}")

    async def run(self):
        """Main orchestration logic for the call."""
        self.call_ongoing = True
        
        # Start the Welcome Message
        asyncio.create_task(self.play_welcome_message())
        
        # Create three concurrent tasks for the main AI loop
        stt_task = asyncio.create_task(self.google_stt_stream())
        llm_task = asyncio.create_task(self.gemini_llm_stream(stt_task.result_queue))
        tts_task = asyncio.create_task(self.google_tts_stream(llm_task.result_queue))

        await asyncio.gather(stt_task, llm_task, tts_task)
        logging.info(f"All tasks for call {self.dialog_id} finished.")

    async def play_welcome_message(self):
        logging.info("Generating and playing welcome message.")
        welcome_text = "Hello! You've reached the AI customer service agent. How can I help you today?"
        
        # Add to conversation history to give Gemini context
        self.conversation_history.append({'role': 'model', 'parts': [welcome_text]})
        
        response = tts_client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=welcome_text),
            voice=texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000
            ),
        )
        # Convert Linear16 to PCMU (G.711 u-law) for RTP
        pcm_audio = response.audio_content[44:] # Skip WAV header
        ulaw_audio = audioop.lin2ulaw(pcm_audio, 2)
        await self.rtp_server.send_audio(ulaw_audio)


    async def google_stt_stream(self):
        """Stream audio from RTP to Google STT and get transcripts."""
        logging.info("Starting Google STT stream...")
        
        async def audio_generator():
            while self.call_ongoing:
                chunk = await self.audio_queue.get()
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        requests = audio_generator()
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000,
                language_code="en-US",
                enable_automatic_punctuation=True,
            ),
            interim_results=False, # Set to False to only get final results
        )

        try:
            responses = speech_client.streaming_recognize(config=streaming_config, requests=requests)
            for response in responses:
                if not response.results:
                    continue
                result = response.results[0]
                if not result.alternatives:
                    continue
                
                transcript = result.alternatives[0].transcript
                if result.is_final:
                    logging.info(f"STT Final Transcript: {transcript}")
                    self.conversation_history.append({'role': 'user', 'parts': [transcript]})
                    # Here you would pass the transcript to the LLM task.
                    # For a more robust system, use another queue here.
                    # For simplicity, we'll imagine a direct handoff in concept.
        except Exception as e:
            logging.error(f"STT streaming error: {e}")
        finally:
            self.call_ongoing = False


    async def gemini_llm_stream(self, transcript_queue):
        """Process transcripts with Gemini."""
        logging.info("Starting Gemini LLM stream...")
        # This part needs to be adapted for a proper queue from STT
        # For now, we will simulate receiving the whole conversation for processing
        while self.call_ongoing:
            await asyncio.sleep(1) # Placeholder to keep the task alive
            # In a real implementation: `transcript = await transcript_queue.get()`
            # And then process it. The current STT implementation doesn't use a queue yet.
            # This is a key area for enhancement to make it fully interactive.


    async def google_tts_stream(self, text_queue):
        """Convert text from LLM to audio and queue for RTP playback."""
        logging.info("Starting Google TTS stream...")
        # In a real implementation: `text = await text_queue.get()`
        while self.call_ongoing:
             await asyncio.sleep(1) # Placeholder


    async def bye(self, dialog):
        """Called when a BYE is received."""
        logging.info(f"Call {self.dialog_id} ended.")
        self.call_ongoing = False
        if self.rtp_server:
            self.rtp_server.close()
        # Signal the STT generator to end
        await self.audio_queue.put(None)


# --- SIP and RTP Handling ---

class SIPApp(AioSIP):
    def __init__(self, loop):
        super().__init__(loop=loop)

    async def on_invite(self, invite):
        dialog = VoiceAgentDialog.from_invite(invite)
        dialog.status_code = 200
        dialog.status_message = "OK"

        # **CRITICAL RTP PART**
        # 1. Create a UDP socket for RTP
        rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rtp_socket.bind(('', 0)) # Bind to an available port
        rtp_host, rtp_port = rtp_socket.getsockname()

        # 2. Tell Telnyx where to send audio in the SDP response
        dialog.sdp = f"""
v=0
o=- 3810058281 3810058281 IN IP4 {VM_STATIC_IP}
s=Talk
c=IN IP4 {VM_STATIC_IP}
t=0 0
m=audio {rtp_port} RTP/AVP 0
a=rtpmap:0 PCMU/8000
a=sendrecv
a=ptime:20
        """.strip()

        await dialog.reply()
        logging.info(f"Replied to INVITE for call {dialog.dialog_id} with SDP for RTP at {VM_STATIC_IP}:{rtp_port}")
        
        # 3. Start the RTP server to handle media
        rtp = Rtp(sock=rtp_socket, loop=self._loop)
        dialog.rtp_server = rtp
        rtp.start(remote_host=invite.sdp['connection']['ip'], remote_port=invite.sdp['media'][0]['port'])
        
        # 4. Start the main orchestration loop for this call
        # This runs in the background for the duration of the call
        asyncio.create_task(self.process_rtp_stream(rtp, dialog))
        asyncio.create_task(dialog.run())

    async def process_rtp_stream(self, rtp, dialog):
        """Receives RTP packets, decodes, and queues them for STT."""
        logging.info("Starting RTP processing loop...")
        while dialog.call_ongoing:
            try:
                # Receive raw RTP packet (header + payload)
                packet = await rtp.recv()
                
                # Payload is PCMU (G.711 u-law) encoded audio
                ulaw_audio = packet.payload
                
                # Convert to Linear16 for Google STT
                # The '2' indicates 2-byte samples (16-bit)
                linear16_audio = audioop.ulaw2lin(ulaw_audio, 2)
                
                # Put the processed audio chunk into the queue for STT
                await dialog.audio_queue.put(linear16_audio)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in RTP processing loop: {e}")
                break
        logging.info("RTP processing loop finished.")
        rtp.close()

# --- Webhook Handler (using aiohttp for simplicity with asyncio) ---

async def handle_telnyx_webhook(request):
    """Handles incoming `call.initiated` webhook from Telnyx."""
    post_data = await request.json()
    event_type = post_data.get('data', {}).get('event_type')

    if event_type == 'call.initiated':
        call_control_id = post_data.get('data', {}).get('payload', {}).get('call_control_id')
        logging.info(f"Received call.initiated for call_control_id: {call_control_id}")

        # This is TeXML, Telnyx's version of TwiML.
        # It instructs Telnyx to connect the call to our SIP endpoint.
        response_xml = f"""
<Response>
    <Connect>
        <Sip>sip:agent@{VM_STATIC_IP}:{SIP_PORT}</Sip>
    </Connect>
</Response>
        """.strip()
        
        return web.Response(text=response_xml, content_type='application/xml')
    
    return web.Response(status=200)

# --- Main Execution Block ---

async def main():
    loop = asyncio.get_running_loop()

    # 1. Start the SIP Server
    sip_app = SIPApp(loop=loop)
    await sip_app.listen(hostname='0.0.0.0', port=SIP_PORT)
    logging.info(f"SIP server listening on 0.0.0.0:{SIP_PORT}")

    # 2. Start the Webhook Server
    webhook_app = web.Application()
    webhook_app.router.add_post('/telnyx-webhook', handle_telnyx_webhook)
    webhook_runner = web.AppRunner(webhook_app)
    await webhook_runner.setup()
    webhook_site = web.TCPSite(webhook_runner, '0.0.0.0', WEBHOOK_PORT)
    await webhook_site.start()
    logging.info(f"Webhook server listening on 0.0.0.0:{WEBHOOK_PORT}")
    
    # Keep servers running
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down.")
Note: The above code is a functional skeleton. A production system would require more robust queueing between STT->LLM->TTS and more sophisticated state management.
3. Dependency Management (requirements.txt)
# requirements.txt

# Web server for webhook and SIP framework core
aiohttp
aiosip

# Google Cloud AI services
google-cloud-speech
google-cloud-texttospeech
google-generativeai

# For loading environment variables
python-dotenv
Install with pip install -r requirements.txt.
4. Deployment Considerations (GCE VM)
GCE VM Setup:
Use a standard image like Debian 11 or Ubuntu 22.04 LTS.
Reserve a Static External IP: This is non-negotiable. Telnyx needs a fixed IP address to send webhooks and SIP traffic to.
Assign an IAM service account to the VM with these roles:Speech-to-Text API User
Text-to-Speech API User
Gemini API User (or a more generic AI Platform role)
Firewall Rules:
In the Google Cloud Console (VPC network -> Firewall), create rules to allow traffic to your VM:
Webhook: Allow Ingress, TCP port 8080 (or 80/443 if using a reverse proxy), Source: Telnyx IP ranges (find these in their documentation, or start with 0.0.0.0/0 for testing).
SIP: Allow Ingress, UDP port 5060, Source: Telnyx SIP IP ranges.
RTP: Allow Ingress, UDP port range (e.g., 10000-20000), Source: Telnyx Media IP ranges. This wide range allows for multiple concurrent calls, as each RTP stream needs its own port.
Process Management (systemd):
To ensure your application runs 24/7 and restarts on failure, create a systemd service file.
sudo nano /etc/systemd/system/voice-ai-agent.service
[Unit]
Description=Voice AI Agent Application
After=network.target

[Service]
User=appuser  # The user you created to run the app
Group=appuser
WorkingDirectory=/home/appuser/voice-ai-agent
# Set GOOGLE_APPLICATION_CREDENTIALS if you are using a key file
# Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/appuser/voice-ai-agent/service-account.json"
EnvironmentFile=/home/appuser/voice-ai-agent/.env
ExecStart=/usr/bin/python3 /home/appuser/voice-ai-agent/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
Enable and start the service:
sudo systemctl daemon-reload
sudo systemctl enable voice-ai-agent.service
sudo systemctl start voice-ai-agent.service
sudo systemctl status voice-ai-agent.service # Check status
journalctl -u voice-ai-agent.service -f # View logs
Environment Variable Setup (.env):
nano /home/appuser/voice-ai-agent/.env
VM_STATIC_IP="YOUR_GCE_STATIC_IP"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# If not using IAM roles on the VM instance itself:
# GOOGLE_APPLICATION_CREDENTIALS="/home/appuser/voice-ai-agent/service-account.json"
5. Telnyx Integration
Create a Call Control Application:
In your Telnyx Mission Control Portal, go to "Call Control" -> "Applications" -> "Add New App".
Give it a name (e.g., "GCP Voice AI Agent").
Set the Webhook URL to http://YOUR_VM_STATIC_IP:8080/telnyx-webhook.
Purchase and Assign a Number:
Go to "Numbers" and buy a phone number.
Edit the number's settings. Under "Connection or App", select your newly created "GCP Voice AI Agent" application.
Now, when you call this number, Telnyx will hit your webhook, your application will respond with the <Connect><Sip>...</Sip></Connect> command, and Telnyx will initiate the SIP call to your agent running on GCE.
Citation Sources

https://github.com/Dungyichao/http_server 
https://github.com/HugoLaurencon/Word-boundaries-information 
https://github.com/dimbambs7/analyzevoice 
https://github.com/googleapis/python-speech 
https://pub.aimind.so/decrypting-words-using-muse-2-c3a3da632219 
https://stackoverflow.com/questions/53290080/displaying-google-cloud-speech-to-text
