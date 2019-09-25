
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using static Microsoft.ML.DataOperationsCatalog;
using System.Linq;
using Microsoft.ML.Data;
using System.IO.Compression;
using System.Threading;
using System.Net;
using Microsoft.ML.Transforms.Image;
using System.Drawing;
using System.Runtime.InteropServices;

namespace Samples.Dynamic
{
    public class ResnetV2101TransferLearningTrainTestSplitBMP
    {
        public static void Example()
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs",
                "imageClassifier.zip");

            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            //Download the image set and unzip
            string finalImagesFolderName = "flower_photos";
            
            //string finalImagesFolderName = DownloadImageSet(
               // imagesDownloadFolderPath);
            
            string fullImagesetFolderPath = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderName);

            try
            {

                MLContext mlContext = new MLContext(seed: 1);
                
                //Load all the original images info
                IEnumerable<ImageData> imagesPathData = LoadImagesFromDirectory(
                    folder: fullImagesetFolderPath, useFolderNameasLabel: true);
                // Get iDataView object from IEnumerable of image data
                var imageView = mlContext.Data.LoadFromEnumerable(imagesPathData);

                // We first shuffle the images before getting the Bitmap Image since ShuffleRows() drops Bitmap type
                //IDataView shuffledFullImagesDataset = imageView;
                IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(imageView);
                shuffledFullImagesDataset = mlContext.Transforms.Conversion
                    .MapValueToKey("Label")
                    .Fit(shuffledFullImagesDataset)
                    .Transform(shuffledFullImagesDataset);
                
                               
                // Split the data 90:10 into train and test sets, train and evaluate.
                TrainTestData trainTestData = mlContext.Data.TrainTestSplit(
                    shuffledFullImagesDataset, testFraction: 0.1, seed: 1);

                IDataView trainDataset = trainTestData.TrainSet;
                IDataView testDataset = trainTestData.TestSet;

                var testDatasetWithImgBuf = mlContext.Transforms.LoadImagesAsBytes("ImageVBuf", fullImagesetFolderPath, "ImagePath")
                    .Fit(testDataset).Transform(testDataset);

                var pipeline = mlContext.Transforms.LoadImagesAsBytes("ImageVBuf", fullImagesetFolderPath, "ImagePath")
                    .Append(mlContext.Model.ImageClassification(
                    "ImageVBuf", "Label",
                    // Just by changing/selecting InceptionV3 here instead of 
                    // ResnetV2101 you can try a different architecture/pre-trained 
                    // model. 
                    arch: ImageClassificationEstimator.Architecture.ResnetV2101,
                    epoch: 50,
                    batchSize: 10,
                    learningRate: 0.01f,
                    metricsCallback: (metrics) => Console.WriteLine(metrics),
                    validationSet: testDatasetWithImgBuf,
                    reuseTrainSetBottleneckCachedValues: false,
                    reuseValidationSetBottleneckCachedValues: false));


                Console.WriteLine("*** Training the image classification model with " +
                    "DNN Transfer Learning on top of the selected pre-trained " +
                    "model/architecture ***");

                // Measuring training time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                var trainedModel = pipeline.Fit(trainDataset);

                watch.Stop();
                long elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Training with transfer learning took: " +
                    (elapsedMs / 1000).ToString() + " seconds");
                /*
                mlContext.Model.Save(trainedModel, testDatasetWithImgObj.Schema,
                    "model.zip");

                ITransformer loadedModel;
                DataViewSchema schema;
                using (var file = File.OpenRead("model.zip"))
                    loadedModel = mlContext.Model.Load(file, out schema);
                
                EvaluateModel(mlContext, testDataset, loadedModel);
                */
                EvaluateModel(mlContext, testDataset, trainedModel);
                VBuffer<ReadOnlyMemory<char>> keys = default;
                trainedModel.GetOutputSchema(testDataset.Schema)["Label"].GetKeyValues(ref keys);

                watch = System.Diagnostics.Stopwatch.StartNew();
                TrySinglePrediction(fullImagesetFolderPath, mlContext, trainedModel,
                    keys.DenseValues().ToArray());

                watch.Stop();
                elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Prediction engine took: " +
                    (elapsedMs / 1000).ToString() + " seconds");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void TrySinglePrediction(string imagesForPredictions,
            MLContext mlContext, ITransformer trainedModel,
            ReadOnlyMemory<char>[] originalLabels)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<ImageData, ImagePrediction>(trainedModel);

            IEnumerable<ImageData> testImages = LoadImagesFromDirectory(
                imagesForPredictions, false);

            //byte[] imgBytes = File.ReadAllBytes(testImages.First().ImagePath);
            //VBuffer<Byte> imgData = new VBuffer<byte>(imgBytes.Length, imgBytes);

            ImageData imageToPredict = new ImageData
            {
                ImagePath = testImages.First().ImagePath,
                //ImageVBuf = imgData


            };

            var prediction = predictionEngine.Predict(imageToPredict);
            var index = prediction.PredictedLabel;

            Console.WriteLine($"ImageFile : " +
                $"[{Path.GetFileName(imageToPredict.ImagePath)}], " +
                $"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {originalLabels[index]}");
        }


        private static void EvaluateModel(MLContext mlContext,
            IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making bulk predictions and evaluating model's " +
                "quality...");

            // Measuring time
            var watch2 = System.Diagnostics.Stopwatch.StartNew();

            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                              $"macro-accuracy = {metrics.MacroAccuracy}");

            watch2.Stop();
            long elapsed2Ms = watch2.ElapsedMilliseconds;

            Console.WriteLine("Predicting and Evaluation took: " +
                (elapsed2Ms / 1000).ToString() + " seconds");
        }

        public static int LoadDataIntoBuffer(string path, ref VBuffer<Byte> imgData)
        {
            int count = -1;
            // bufferSize == 1 used to avoid unnecessary buffer in FileStream
            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1))
            {
                long fileLength = fs.Length;
                if (fileLength > int.MaxValue)
                    throw new IOException($"File {path} too big to open.");
                else if (fileLength == 0)
                {
                    byte[] _imageBuffer;                  

                    // Some file systems (e.g. procfs on Linux) return 0 for length even when there's content.
                    // Thus we need to assume 0 doesn't mean empty.
                    _imageBuffer = File.ReadAllBytes(path);
                    count = _imageBuffer.Length;
                    Console.WriteLine("File length is zero");
                }

                count = (int)fileLength;
                var editor = VBufferEditor.Create(ref imgData, count);
                //var buffer = File.ReadAllBytes(path);
                fs.Read(editor.Values);
                /*
                for (int i = 0; i < count; i++)
                {
                    //editor.Values[i] = (byte) fs.ReadByte();
                    editor.Values[i] = buffer[i];
                }
                */
                imgData = editor.Commit();

                return count;
            }
        }


        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder,
            bool useFolderNameasLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            //VBuffer<Byte> imgData = new VBuffer<byte>();
            
            foreach (var file in files)
            {
                if (Path.GetExtension(file) != ".JPEG" && Path.GetExtension(file) != ".jpg")
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                // Get the buffer of bytes
                //int imgSize = LoadDataIntoBuffer(file, ref imgData);


                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label,
                    //ImageVBuf = imgData
                };

            }
        }

        /*
        public static IEnumerable<ImageDataBMP> LoadImagesFromDirectoryToBitmap(string folder,
            bool useFolderNameasLabel = true)
        {

            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if (Path.GetExtension(file) != ".jpg")
                    continue;

                // Get label for the image
                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                // Load image into a bitmap object
                

                yield return new ImageDataBMP()
                {
                    Image = mlContext.Transforms.LoadImages("ImageObject",
                files, "ImagePath"),
                    Label = label
                };

            }

        }
        */

        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes

            //SINGLE SMALL FLOWERS IMAGESET (200 files)
            
            string fileName = "flower_photos_small_set.zip";
            
            string url = $"https://mlnetfilestorage.file.core.windows.net/" +
                $"imagesets/flower_images/flower_photos_small_set.zip?st=2019-08-" +
                $"07T21%3A27%3A44Z&se=2030-08-08T21%3A27%3A00Z&sp=rl&sv=2018-03-" +
                $"28&sr=f&sig=SZ0UBX47pXD0F1rmrOM%2BfcwbPVob8hlgFtIlN89micM%3D";
            
            //string fileName = "flower_photos.tgz";

            //string url = $"http://download.tensorflow.org/example_images/flower_photos.tgz";

            Download(url, imagesDownloadFolder, fileName);
            UnZip(Path.Combine(imagesDownloadFolder, fileName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        public static bool Download(string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = url.Split(Path.DirectorySeparatorChar).Last();

            Directory.CreateDirectory(destDir);

            string relativeFilePath = Path.Combine(destDir, destFileName);

            if (File.Exists(relativeFilePath))
            {
                Console.WriteLine($"{relativeFilePath} already exists.");
                return false;
            }

            var wc = new WebClient();
            Console.WriteLine($"Downloading {relativeFilePath}");
            var download = Task.Run(() => wc.DownloadFile(url, relativeFilePath));
            while (!download.IsCompleted)
            {
                Thread.Sleep(1000);
                Console.Write(".");
            }
            Console.WriteLine("");
            Console.WriteLine($"Downloaded {relativeFilePath}");

            return true;
        }

        public static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar)
                .Last()
                .Split('.')
                .First() + ".bin";

            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                ZipFile.ExtractToDirectory(gzArchiveName, destFolder);
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(
                ResnetV2101TransferLearningTrainTestSplit).Assembly.Location);

            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }


        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
                        
            //public VBuffer<byte> ImageVBuf;

        }


        private static IEnumerable<ImageData> GetImageDataLabels(IDataView transformedData, bool useFolderNameasLabel = true)
        {
            // The transformedData IDataView contains the loaded images now.
            // Console.WriteLine("{0, -25} {1, -25} {2, -25}", "ImagePath", "Name",
            //    "ImageObject");

            using (var cursor = transformedData.GetRowCursor(transformedData
                .Schema))
            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to faciliate buffer sharing (if applicable),
                // and column-type validation once, rather than many times.
                ReadOnlyMemory<char> imagePath = default;
                ReadOnlyMemory<char> name = default;
                Bitmap imageObject = null;

                var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["ImagePath"]);

                var nameGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["Name"]);

                var imageObjectGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                    "ImageObject"]);

                while (cursor.MoveNext())
                {

                    imagePathGetter(ref imagePath);
                    nameGetter(ref name);
                    imageObjectGetter(ref imageObject);

                    //get label
                    string imgPath = imagePath.ToString();
                    var label = Path.GetFileName(imgPath);
                    if (useFolderNameasLabel)
                        label = Directory.GetParent(imgPath).Name;
                    else
                    {
                        for (int index = 0; index < label.Length; index++)
                        {
                            if (!char.IsLetter(label[index]))
                            {
                                label = label.Substring(0, index);
                                break;
                            }
                        }
                    }


                    Console.WriteLine("{0, -25} {1, -25} {2, -25}", imagePath, name,
                        imageObject.PhysicalDimension);

                    yield return new ImageData()
                    {
                        ImagePath = imgPath,
                        Label = label,
                        
                    };
                }

                // Dispose the image.
                //imageObject.Dispose();
            }
        }

        /*
        
        private class ImageDataBMP
        {
            [ImageType(3, 4)]
            public Bitmap Image { get; set; }

            [ImageType(3, 4)]
            public Bitmap GrayImage { get; set; }

            //[ColumnName("label")]
            public string Label;


            public ImageDataBMP()
            {
                Image = null;
                GrayImage = null;
            }

            public ImageDataBMP(int width, int height, Color color)
            {
                Image = new Bitmap(width, height);
                for (int i = 0; i < width; ++i)
                    for (int j = 0; j < height; ++j)
                        Image.SetPixel(i, j, color);
            }
        }
        */
        public class ImagePrediction
        {
            [ColumnName("Score")]
            public float[] Score;

            [ColumnName("PredictedLabel")]
            public UInt32 PredictedLabel;
        }
    }
}

