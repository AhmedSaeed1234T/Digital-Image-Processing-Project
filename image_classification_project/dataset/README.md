# Dataset Instructions

## Adding Person/Human Images for Training

To train the classifier to recognize humans/persons:

1. **Create a "person" folder** (if it doesn't exist):
   - The app will automatically create this folder on first run
   - Location: `image_classification_project/dataset/person/`

2. **Add person images**:
   - Add at least 2-3 images of people/humans to the `person` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
   - Name them: `person1.jpg`, `person2.jpg`, etc.

3. **Restart the app**:
   - Close and reopen the application
   - The classifier will automatically retrain with the new person images

## Current Classes

The classifier is currently trained on:
- **cats** - Images of cats
- **cars** - Images of cars
- **person** - Images of humans/people (add images here)

## Tips for Better Classification

- Add multiple images per class (at least 2-3 images minimum)
- Use clear, well-lit images
- Ensure images show the main subject clearly
- More training images = better accuracy

