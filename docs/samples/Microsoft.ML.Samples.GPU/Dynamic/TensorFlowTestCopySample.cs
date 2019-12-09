using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public class TensorFlowTestCopySample
    {
        public static string parentWorkspacePath;
        public static string assetsPath;

        internal static void CreateParentWorkspacePathForImageClassification()
        {
            string assetsRelativePath = @"assets";
            assetsPath = GetAbsolutePath(assetsRelativePath);
            string workspacePath = Path.Combine(assetsPath, "cached");
            // Delete if the workspace path already exists
            if (Directory.Exists(workspacePath))
            {
                Directory.Delete(workspacePath, true);
            }

            // Create a new empty workspace path
            Directory.CreateDirectory(workspacePath);
            parentWorkspacePath = workspacePath;
        }

        public TensorFlowTestCopySample()
        {
            CreateParentWorkspacePathForImageClassification();
        }

        public static void Example()
        {
            CreateParentWorkspacePathForImageClassification();
            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs",
                "images");

            //Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(
                imagesDownloadFolderPath);

            string fullImagesetFolderPath = Path.Combine(
                imagesDownloadFolderPath, finalImagesFolderName);

            MLContext mlContext = new MLContext(seed: 1);

            //Load all the original images info
            IEnumerable<ImageData> images = LoadImagesFromDirectory(
                folder: fullImagesetFolderPath, useFolderNameAsLabel: true);

            IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(
                mlContext.Data.LoadFromEnumerable(images), seed: 1);

            shuffledFullImagesDataset = mlContext.Transforms.Conversion
                .MapValueToKey("Label")
                .Fit(shuffledFullImagesDataset)
                .Transform(shuffledFullImagesDataset);

            // Split the data 80:20 into train and test sets, train and evaluate.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(
                shuffledFullImagesDataset, testFraction: 0.2, seed: 1);

            IDataView trainDataset = trainTestData.TrainSet;
            IDataView testDataset = trainTestData.TestSet;
            var validationSet = mlContext.Transforms.LoadRawImageBytes("Image", fullImagesetFolderPath, "ImagePath")
                    .Fit(testDataset)
                    .Transform(testDataset);

            var arch = ImageClassificationTrainer.Architecture.ResnetV250All;
            // Check if the bottleneck cached values already exist
            var (trainSetBottleneckCachedValuesFileName, validationSetBottleneckCachedValuesFileName,
                workspacePath, isReuse) = getInitialParameters(arch, finalImagesFolderName);

            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "Label",
                // Just by changing/selecting InceptionV3/MobilenetV2 here instead of 
                // ResnetV2101 you can try a different architecture/
                // pre-trained model. 
                Arch = arch,
                Epoch = 5,
                BatchSize = 10,
                LearningRate = 0.01f,
                MetricsCallback = (metric) => Console.WriteLine(metric),
                TestOnTrainSet = false,
                WorkspacePath = workspacePath,
                ReuseTrainSetBottleneckCachedValues = isReuse,
                ReuseValidationSetBottleneckCachedValues = isReuse,
                TrainSetBottleneckCachedValuesFileName = trainSetBottleneckCachedValuesFileName,
                ValidationSetBottleneckCachedValuesFileName = validationSetBottleneckCachedValuesFileName,
                ValidationSet = validationSet,
                IsTrainAllLayers = arch == ImageClassificationTrainer.Architecture.ResnetV250All,
                Layers = new string[] { "final_retrain_ops" }
            };

            var pipeline = mlContext.Transforms.LoadRawImageBytes("Image", fullImagesetFolderPath, "ImagePath")
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel")));

            var trainedModel = pipeline.Fit(trainDataset);

            mlContext.Model.Save(trainedModel, shuffledFullImagesDataset.Schema,
                "model.zip");

            ITransformer loadedModel;
            DataViewSchema schema;
            using (var file = File.OpenRead("model.zip"))
                loadedModel = mlContext.Model.Load(file, out schema);

            // Testing EvaluateModel: group testing on test dataset
            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            //Assert.InRange(metrics.MicroAccuracy, 0.8, 1);
            //Assert.InRange(metrics.MacroAccuracy, 0.8, 1);

            // Testing TrySinglePrediction: Utilizing PredictionEngine for single
            // predictions. Here, two pre-selected images are utilized in testing
            // the Prediction engine.
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

            IEnumerable<ImageData> testImages = LoadImagesFromDirectory(
                fullImagesetFolderPath, true);

            string[] directories = Directory.GetDirectories(fullImagesetFolderPath);
            string[] labels = new string[directories.Length];
            for (int j = 0; j < labels.Length; j++)
            {
                var dir = new DirectoryInfo(directories[j]);
                labels[j] = dir.Name;
            }

            // Test daisy image
            ImageData firstImageToPredict = new ImageData
            {
                ImagePath = Path.Combine(fullImagesetFolderPath, "daisy", "5794835_d15905c7c8_n.jpg")
            };

            // Test rose image
            ImageData secondImageToPredict = new ImageData
            {
                ImagePath = Path.Combine(fullImagesetFolderPath, "roses", "12240303_80d87f77a3_n.jpg")
            };

            var predictionFirst = predictionEngine.Predict(firstImageToPredict);
            var predictionSecond = predictionEngine.Predict(secondImageToPredict);
        }

        internal static bool ShouldReuse(string workspacePath, string trainSetBottleneckCachedValuesFileName, string validationSetBottleneckCachedValuesFileName)
        {
            bool isReuse = false;
            if (Directory.Exists(workspacePath) && File.Exists(Path.Combine(workspacePath, trainSetBottleneckCachedValuesFileName))
                && File.Exists(Path.Combine(workspacePath, validationSetBottleneckCachedValuesFileName)))
            {
                isReuse = true;
            }
            else
            {
                Directory.CreateDirectory(workspacePath);
            }
            return isReuse;
        }

        internal static (string, string, string, bool) getInitialParameters(ImageClassificationTrainer.Architecture arch, string finalImagesFolderName)
        {
            string trainSetBottleneckCachedValuesFileName = "TrainsetCached_" + finalImagesFolderName + "_" + (int)arch;
            string validationSetBottleneckCachedValuesFileName = "validationsetCached_" + finalImagesFolderName + "_" + (int)arch;
            string workspacePath = Path.Combine(parentWorkspacePath, finalImagesFolderName + "_" + (int)arch);
            bool isReuse = ShouldReuse(workspacePath, trainSetBottleneckCachedValuesFileName, validationSetBottleneckCachedValuesFileName);
            return (trainSetBottleneckCachedValuesFileName, validationSetBottleneckCachedValuesFileName, workspacePath, isReuse);
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder,
            bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);
            /*
             * This is only needed as Linux can produce files in a different 
             * order than other OSes. As this is a test case we want to maintain
             * consistent accuracy across all OSes, so we sort to remove this discrepency.
             */
            Array.Sort(files);
            foreach (var file in files)
            {
                if (Path.GetExtension(file) != ".jpg")
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameAsLabel)
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

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };

            }
        }

        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            string fileName = "flower_photos_tiny_set_for_unit_tests.zip";
            string filenameAlias = "FPTSUT"; // FPTSUT = flower photos tiny set for unit tests
            string url = "https://aka.ms/mlnet-resources/flower_photos_tiny_set_for_unit_test.zip";

            Download(url, imagesDownloadFolder, fileName);
            UnZip(Path.Combine(imagesDownloadFolder, fileName), imagesDownloadFolder);
            // Sometimes tests fail because the path is too long. So rename the dataset folder to a shorter directory.
            if (!Directory.Exists(Path.Combine(imagesDownloadFolder, filenameAlias)))
                Directory.Move(Path.Combine(imagesDownloadFolder, Path.GetFileNameWithoutExtension(fileName)), Path.Combine(imagesDownloadFolder, "FPTSUT"));
            return filenameAlias;
        }

        public static string DownloadBadImageSet(string imagesDownloadFolder)
        {
            string fileName = "CatsVsDogs_tiny_for_unit_tests.zip";
            string url = $"https://tlcresources.blob.core.windows.net/datasets/" +
                $"CatsVsDogs_tiny_for_unit_tests.zip";

            Download(url, imagesDownloadFolder, fileName);
            UnZip(Path.Combine(imagesDownloadFolder, fileName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        private static bool Download(string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = url.Split(Path.DirectorySeparatorChar).Last();

            Directory.CreateDirectory(destDir);

            string relativeFilePath = Path.Combine(destDir, destFileName);

            if (File.Exists(relativeFilePath))
                return false;

            new WebClient().DownloadFile(url, relativeFilePath);
            return true;
        }

        private static void UnZip(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar)
                .Last()
                .Split('.')
                .First() + ".bin";

            if (File.Exists(Path.Combine(destFolder, flag)))
                return;

            ZipFile.ExtractToDirectory(gzArchiveName, destFolder);
            File.Create(Path.Combine(destFolder, flag));
        }

        public static string GetAbsolutePath(string relativePath) =>
            Path.Combine(new FileInfo(typeof(
                TensorFlowTestCopySample).Assembly.Location).Directory.FullName, relativePath);


        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }

        public class ImagePrediction
        {
            [ColumnName("Score")]
            public float[] Score;

            [ColumnName("PredictedLabel")]
            public string PredictedLabel;
        }

        private static string GetTemporaryDirectory()
        {
            string tempDirectory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            Directory.CreateDirectory(tempDirectory);
            return tempDirectory;
        }

    }
}
