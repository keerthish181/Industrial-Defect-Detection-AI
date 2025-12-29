SteelSafe AI is an AI-powered industrial defect detection platform designed to automate quality inspection processes in manufacturing and steel industries. The system leverages computer vision and deep learning to analyze product surface images and accurately classify them as Normal or Defective, reducing reliance on manual inspection and improving consistency, speed, and accuracy.

The application is built using a Convolutional Neural Network (CNN) with transfer learning, implemented using TensorFlow and Keras, enabling reliable defect detection even with limited training data. Uploaded images undergo preprocessing and quality validation checks such as blur and brightness detection before being analyzed by the model. The system provides confidence scores for every prediction, allowing adjustable sensitivity through a configurable threshold.

SteelSafe AI includes a Flask-based web interface that supports both operator and supervisor views. Operators can upload single or batch images for inspection, while supervisors can monitor inspection summaries, statistics, and recent activity. The platform maintains an inspection history, displays real-time statistics, and visualizes results using interactive graphs for better decision-making.

To enhance transparency and trust in AI predictions, the system incorporates Grad-CAM heatmap visualization, highlighting the regions of the image that influenced defect detection. Additionally, the platform generates automated PDF inspection reports, capturing key details such as inspection time, result, confidence level, and defect type.

SteelSafe AI demonstrates a real-world, enterprise-ready approach to automated quality control, showcasing the integration of machine learning, image processing, and web technologies. The project is scalable and suitable for future enhancements such as multi-class defect detection, real-time camera integration, and cloud-based deployment.

## Dataset
Due to GitHub file size limitations, the dataset is not included in this repository.

The model was trained using publicly available industrial surface defect datasets such as:
- NEU Surface Defect Dataset
- Kaggle Metal Surface Defect Dataset

Users can download the dataset from the original sources and place images in the required directory structure.
