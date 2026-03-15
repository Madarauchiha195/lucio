# How to Run Lucio Challenge on Google Colab

Processing 265,000+ chunks takes 1–2 hours on a regular laptop CPU. Google Colab provides powerful cloud CPUs (and GPUs) that can speed this up. Follow these exact steps to run the pipeline in Colab.

### Step 1: Zip the Project Folder
Before moving to Colab, you need to compress this entire `d:\lucio` folder.
1. Open Windows File Explorer.
2. Right-click the `d:\lucio` folder.
3. Select **Compress to ZIP file** (or "Send to" > "Compressed (zipped) folder").
4. Name it `lucio.zip`.

### Step 2: Open Google Colab
1. Go to [https://colab.research.google.com/](https://colab.research.google.com/) and sign in with a Google account.
2. Click **New Notebook**.
3. (Optional but recommended) In the top menu, go to **Runtime** > **Change runtime type**, and select **T4 GPU** to speed up FAISS and SentenceTransformers.

### Step 3: Upload the Zip File
1. In the Colab sidebar (on the left), click the **Folder icon** 📁 to open the Files panel.
2. Click the **Upload** icon (a piece of paper with an up arrow) and select your `lucio.zip` file.
3. Wait for the upload to complete (it may take a few minutes since you have many PDF documents inside).

### Step 4: Run the Extraction and Setup
In your Colab notebook, copy and paste the following commands into a code cell, then press the Play button (or `Shift + Enter`):

```python
# 1. Unzip the project folder
!unzip -q lucio.zip -d lucio

# 2. Change directory into the project
%cd lucio

# 3. Install necessary dependencies
!pip install -r requirements.txt
!pip install google-genai openpyxl pandas
```

### Step 5: Verify Your API Key
Make sure your `.env` file successfully carried over. Run this in a new cell:
```python
!cat .env
```
*(If it's missing or empty, you can manually recreate it by running `!echo "GEMINI_API_KEY=AIzaSyAzdnKN31Y-OiU0Nrn5HWJ_DlaYeT-xAJU" > .env`)*

### Step 6: Run the AI Pipeline!
Finally, run the `submit.py` script. This will chew through the 265,000 chunks, build the AI vectors natively on Google's fast infrastructure, answer the Excel questions, and save the final `lucio_submission.json`.

Run this in a new cell:
```python
!python submit.py
```

### Step 7: Download Your Results
Once it finishes, look back at the left-hand folder sidebar. 
1. Open the `lucio` folder in the sidebar.
2. Find the newly generated `lucio_submission.json` file.
3. Click the three dots `⋮` next to it, and select **Download**. You can now submit this file to the challenge organizers!
