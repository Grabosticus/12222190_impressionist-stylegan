# Demo Application
This demo application lets you generate your own impressionist artworks by calling a FastAPI backend, that runs my StyleGAN model. Since I was only able to train the StyleGAN up to a resolution of 64x64, I included an upsampling functionality in the backend, which uses an AI to increase the image resolution by a factor of 4.

## Running the application

### Running the backend
```
cd backend
conda env create -f environment_demo_app.yml
conda activate demo_application
uvicorn main:app --port 8000
```

Then wait for the backend to fully start up. You know that the backend is now functionable, when you see the following output in your terminal:
```
Initializing generator...
Generator initialization complete!
Initializing upsampler
Upsampler initialization complete
INFO:     Started server process [36190]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Running the frontend
```
cd frontend
npm i
npm run dev
```

Then you can generate and upsample your own images using the button in the frontend application.

**IMPORTANT**
Although the amount of upsampling steps you can perform is hypothetically unliminted, you will encounter thresholds based on your hardware capabilities pretty fast. On my MacBook M1, I was able to upsample images up to 1024x1024.
  