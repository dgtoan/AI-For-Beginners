## <a href="README.md">English</a> | <a href="README.vi.md">Tiếng Việt</a>

# Giáo Trình - Artificial Intelligence for Beginners

|![ Sketchnote by [(@girlie_mac)](https://twitter.com/girlie_mac) ](./lessons/sketchnotes/ai-overview.png)|
|:---:|
| AI For Beginners - _Sketchnote by [@girlie_mac](https://twitter.com/girlie_mac)_ |

Khóa học **Trí tuệ Nhân tạo (AI)** trong 12 tuần, 24 bài học cùng với Toanr!

Trong chương trình giảng dạy này, bạn sẽ học được:

- Các phương pháp tiếp cận khác nhau đối với Trí tuệ Nhân tạo, bao gồm cả phương pháp "cổ điển" ([GOFAI](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence) - [GOFAI_vi](https://www.phamduytung.com/blog/2023-02-18-symbolic-vs-connectionist/)).
- **Neural Networks** và **Deep Learning**, cốt lõi của AI hiện đại. Các khái niệm và chủ đề được trình bày bằng 2 frameworks - [TensorFlow](http://Tensorflow.org) and [PyTorch](http://pytorch.org).
- **Kiến trúc Nơ-ron (Neural Architectures)** để xử lý bài toán về hình ảnh và văn bản. Khóa học có đề cập đến đến các mô hình gần đây nhưng không hẳn là tiên tiến nhất.
- Một số phương pháp sử dụng AI ít phổ biến hơn, chẳng hạn như **Thuật toán Di truyền (Genetic Algorithms)** và **Hệ thống Đa tác nhân (Multi-Agent Systems)**.
- ...

Những thứ không được đề cập trong chương trình này:

- **Học Máy Cổ điển (Machine Learning)** và Toán học Sâu đằng sau Deep Learning - [tham khảo](https://machinelearningcoban.com/).
- **AI Đối thoại và Chatbot.**
- ...


---
# Nội Dung

<table>
<tr><th>No</th><th>Lesson</th><th>Intro</th><th>PyTorch</th><th>Keras/TensorFlow</th><th>Lab</th></tr>

<tr><td>I</td><td colspan="4"><b>Introduction to AI</b></td><td></td></tr>
<tr><td>1</td><td>Introduction and History of AI</td><td><a href="lessons/1-Intro/README.md">Text</a></td><td></td><td></td><td></td></tr>

<tr><td>II</td><td colspan="4"><b>Symbolic AI</b></td><td></td></tr>
<tr><td>2 </td><td>Knowledge Representation and Expert Systems</td><td><a href="lessons/2-Symbolic/README.md">Text</a></td><td colspan="2"><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/2-Symbolic/Animals.ipynb">Expert System</a>, <a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/2-Symbolic/FamilyOntology.ipynb">Ontology</a>, <a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/2-Symbolic/MSConceptGraph.ipynb">Concept Graph</a></td><td></td></tr>
<tr><td>III</td><td colspan="4"><b><a href="lessons/3-NeuralNetworks/README.md">Introduction to Neural Networks</a></b></td><td></td></tr>
<tr><td>3</td><td>Perceptron</td>
   <td><a href="lessons/3-NeuralNetworks/03-Perceptron/README.md">Text</a>
   <td colspan="2"><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/3-NeuralNetworks/03-Perceptron/Perceptron.ipynb">Notebook</a></td><td><a href="lessons/3-NeuralNetworks/03-Perceptron/lab/README.md">Lab</a></td></tr>
<tr><td>4 </td><td>Multi-Layered Perceptron and Creating our own Framework</td><td><a href="lessons/3-NeuralNetworks/04-OwnFramework/README.md">Text</a></td><td colspan="2"><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/3-NeuralNetworks/04-OwnFramework/OwnFramework.ipynb">Notebook</a><td><a href="lessons/3-NeuralNetworks/04-OwnFramework/lab/README.md">Lab</a></td></tr> 
<tr><td>5</td>
   <td>Intro to Frameworks (PyTorch/TensorFlow) and Overfitting</td>
   <td><a href="lessons/3-NeuralNetworks/05-Frameworks/README.md">Text</a></td>
   <td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/3-NeuralNetworks/05-Frameworks/IntroPyTorch.ipynb">PyTorch</a></td>
   <td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/3-NeuralNetworks/05-Frameworks/IntroKeras.ipynb">Keras</a>/<a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/3-NeuralNetworks/05-Frameworks/IntroKerasTF.ipynb">TensorFlow</a></td>
   <td><a href="lessons/3-NeuralNetworks/05-Frameworks/lab/README.md">Lab</a></td></tr>
<tr><td>IV</td><td><b><a href="lessons/4-ComputerVision/README.md">Computer Vision</a></b></td>
  <td colspan="3"><a href="https://docs.microsoft.com/learn/paths/explore-computer-vision-microsoft-azure/?WT.mc_id=academic-77998-cacaste"><i>Microsoft Azure AI Fundamentals: Explore Computer Vision</i></a></td>
  <td></td></tr>
<tr><td></td><td colspan="2"><i>Microsoft Learn Module on Computer Vision</i></td>
  <td><a href="https://docs.microsoft.com/learn/modules/intro-computer-vision-pytorch/?WT.mc_id=academic-77998-cacaste"><i>PyTorch</i></a></td>
  <td><a href="https://docs.microsoft.com/learn/modules/intro-computer-vision-TensorFlow/?WT.mc_id=academic-77998-cacaste"><i>TensorFlow</i></a></td>
  <td></td></tr>
<tr><td>6</td><td>Intro to Computer Vision. OpenCV</td><td><a href="lessons/4-ComputerVision/06-IntroCV/README.md">Text</a><td colspan="2"><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/06-IntroCV/OpenCV.ipynb">Notebook</a></td><td><a href="lessons/4-ComputerVision/06-IntroCV/lab/README.md">Lab</a></td></tr>
<tr><td>7</td><td>Convolutional Neural Networks<br/>CNN Architectures</td><td><a href="lessons/4-ComputerVision/07-ConvNets/README.md">Text</a><br/><a href="lessons/4-ComputerVision/07-ConvNets/CNN_Architectures.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/07-ConvNets/ConvNetsPyTorch.ipynb">PyTorch</a></td><td><a href="lessons/4-ComputerVision/07-ConvNets/ConvNetsTF.ipynb">TensorFlow</a></td><td><a href="lessons/4-ComputerVision/07-ConvNets/lab/README.md">Lab</a></td></tr>
<tr><td>8</td><td>Pre-trained Networks and Transfer Learning<br/>Training Tricks</td><td><a href="lessons/4-ComputerVision/08-TransferLearning/README.md">Text</a><br/><a href="lessons/4-ComputerVision/08-TransferLearning/TrainingTricks.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/08-TransferLearning/TransferLearningPyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/08-TransferLearning/TransferLearningTF.ipynb">TensorFlow</a><br/><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/08-TransferLearning/Dropout.ipynb">Dropout sample</a><br/><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/08-TransferLearning/AdversarialCat_TF.ipynb">Adversarial Cat</a></td><td><a href="lessons/4-ComputerVision/08-TransferLearning/lab/README.md">Lab</a></td></tr>
<tr><td>9</td><td>Autoencoders and VAEs</td><td><a href="lessons/4-ComputerVision/09-Autoencoders/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/09-Autoencoders/AutoEncodersPyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/09-Autoencoders/AutoencodersTF.ipynb">TensorFlow</a></td><td></td></tr>
<tr><td>10</td><td>Generative Adversarial Networks<br/>Artistic Style Transfer</td><td><a href="lessons/4-ComputerVision/10-GANs/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/10-GANs/GANPyTorch.ipynb">PyTorch</td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/10-GANs/GANTF.ipynb">TensorFlow GAN</a><br/><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/10-GANs/StyleTransfer.ipynb">Style Transfer</a></td><td></td></tr>
<tr><td>11</td><td>Object Detection</td><td><a href="lessons/4-ComputerVision/11-ObjectDetection/README.md">Text</a></td><td>PyTorch</td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/11-ObjectDetection/ObjectDetection.ipynb">TensorFlow</td><td><a href="lessons/4-ComputerVision/11-ObjectDetection/lab/README.md">Lab</a></td></tr>
<tr><td>12</td><td>Semantic Segmentation. U-Net</td><td><a href="lessons/4-ComputerVision/12-Segmentation/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/4-ComputerVision/12-Segmentation/SemanticSegmentationPytorch.ipynb">PyTorch</td><td><a href="lessons/4-ComputerVision/12-Segmentation/SemanticSegmentationTF.ipynb">TensorFlow</td><td></td></tr>
<tr><td>V</td><td><b><a href="lessons/5-NLP/README.md">Natural Language Processing</a></b></td>
   <td colspan="3"><a href="https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77998-cacaste"><i>Microsoft Azure AI Fundamentals: Explore Natural Language Processing</i></a></td>
   <td></td></tr>
<tr><td></td><td colspan="2"><i>Microsoft Learn Module on Natural language processing</i></td>
   <td><a href="https://docs.microsoft.com/learn/modules/intro-natural-language-processing-pytorch/?WT.mc_id=academic-77998-cacaste"><i>PyTorch</i></a></td>
   <td><a href="https://docs.microsoft.com/learn/modules/intro-natural-language-processing-TensorFlow/?WT.mc_id=academic-77998-cacaste"><i>TensorFlow</i></a></td>
   <td></td></tr>
<tr><td>13</td><td>Text Representation. Bow/TF-IDF</td><td><a href="lessons/5-NLP/13-TextRep/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/13-TextRep/TextRepresentationPyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/13-TextRep/TextRepresentationTF.ipynb">TensorFlow</td><td></td></tr>
<tr><td>14</td><td>Semantic word embeddings. Word2Vec and GloVe</td><td><a href="lessons/5-NLP/14-Embeddings/README.md">Text</td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/14-Embeddings/EmbeddingsPyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/14-Embeddings/EmbeddingsTF.ipynb">TensorFlow</a></td><td></td></tr>
<tr><td>15</td><td>Language Modeling. Training your own embeddings</td><td><a href="lessons/5-NLP/15-LanguageModeling/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/15-LanguageModeling/CBoW-PyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/15-LanguageModeling/CBoW-TF.ipynb">TensorFlow</a></td><td><a href="lessons/5-NLP/15-LanguageModeling/lab/README.md">Lab</a></td></tr>
<tr><td>16</td><td>Recurrent Neural Networks</td><td><a href="lessons/5-NLP/16-RNN/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/16-RNN/RNNPyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/16-RNN/RNNTF.ipynb">TensorFlow</a></td><td></td></tr>
<tr><td>17</td><td>Generative Recurrent Networks</td><td><a href="lessons/5-NLP/17-GenerativeNetworks/README.md">Text</a></td><td><a href="lessons/5-NLP/17-GenerativeNetworks/GenerativePyTorch.md">PyTorch</a></td><td><a href="lessons/5-NLP/17-GenerativeNetworks/GenerativeTF.md">TensorFlow</a></td><td><a href="lessons/5-NLP/17-GenerativeNetworks/lab/README.md">Lab</a></td></tr>
<tr><td>18</td><td>Transformers. BERT.</td><td><a href="lessons/5-NLP/18-Transformers/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/18-Transformers/TransformersPyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/5-NLP/18-Transformers/TransformersTF.ipynb">TensorFlow</a></td><td></td></tr>
<tr><td>19</td><td>Named Entity Recognition</td><td><a href="lessons/5-NLP/19-NER/README.md">Text</a></td><td></td><td><a href="lessons/5-NLP/19-NER/NER-TF.ipynb">TensorFlow</a></td><td><a href="lessons/5-NLP/19-NER/lab/README.md">Lab</a></td></tr>
<tr><td>20</td><td>Large Language Models, Prompt Programming and Few-Shot Tasks</td><td><a href="lessons/5-NLP/20-LangModels/README.md">Text</a></td><td><a href="lessons/5-NLP/20-LangModels/GPT-PyTorch.ipynb">PyTorch</td><td></td><td></td></tr>
<tr><td>VI</td><td colspan="4"><b>Other AI Techniques</b></td><td></td></tr>
<tr><td>21</td><td>Genetic Algorithms</td><td><a href="lessons/6-Other/21-GeneticAlgorithms/README.md">Text</a><td colspan="2"><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/6-Other/21-GeneticAlgorithms/Genetic.ipynb">Notebook</a></td><td></td></tr>
<tr><td>22</td><td>Deep Reinforcement Learning</td><td><a href="lessons/6-Other/22-DeepRL/README.md">Text</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/6-Other/22-DeepRL/CartPole-RL-PyTorch.ipynb">PyTorch</a></td><td><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/6-Other/22-DeepRL/CartPole-RL-TF.ipynb">TensorFlow</a></td><td><a href="lessons/6-Other/22-DeepRL/lab/README.md">Lab</a></td></tr>
<tr><td>23</td><td>Multi-Agent Systems</td><td><a href="lessons/6-Other/23-MultiagentSystems/README.md">Text</a></td><td></td><td></td><td></td></tr>
<tr><td>VII</td><td colspan="4"><b>AI Ethics</b></td><td></td></tr>
<tr><td>24</td><td>AI Ethics and Responsible AI</td><td><a href="lessons/7-Ethics/README.md">Text</a></td><td colspan="2"><a href="https://docs.microsoft.com/learn/paths/responsible-ai-business-principles/?WT.mc_id=academic-77998-cacaste"><i>MS Learn: Responsible AI Principles</i></a></td><td></td></tr>
<tr><td></td><td colspan="4"><b>Extras</b></td><td></td></tr>
<tr><td>X1</td><td>Multi-Modal Networks, CLIP and VQGAN</td><td><a href="lessons/X-Extras/X1-MultiModal/README.md">Text</a></td><td colspan="2"><a href="https://github.com/microsoft/AI-For-Beginners/blob/main/lessons/X-Extras/X1-MultiModal/Clip.ipynb">Notebook</a></td><td></td></tr>
</table>

**[Mindmap of the Course](http://soshnikov.com/courses/ai-for-beginners/mindmap.html)**

Mỗi bài học chứa tài liệu đọc trước và một Jupyter Notebook chứa các lệnh trong bài đồng thời cũng chứa rất nhiều lý thuyết, vì vậy để hiểu bài học, bạn hãy xem ít nhất một bản Notebook (PyTorch hoặc TensorFlow). Ngoài ra còn có Lab dành cho một số bài học, giúp áp dụng tài liệu đã học vào một vấn đề cụ thể.

# Một Số Tài Liệu Tham Khảo Khác

- [Blog Machine Learning cơ bản](https://machinelearningcoban.com/)
- [Blog Deep Learning cơ bản](https://nttuan8.com/category/deep-learning/)
- [Blog Đắm mình vào học sâu](https://d2l.aivivn.com/)
- ...
