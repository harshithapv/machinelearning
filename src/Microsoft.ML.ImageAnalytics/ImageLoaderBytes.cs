// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;

[assembly: LoadableClass(ImageLoadingTransformerByte.Summary, typeof(IDataTransform), typeof(ImageLoadingTransformerByte), typeof(ImageLoadingTransformerByte.Options), typeof(SignatureDataTransform),
    ImageLoadingTransformerByte.UserName, "ImageLoaderTransform", "ImageLoader")]

[assembly: LoadableClass(ImageLoadingTransformerByte.Summary, typeof(IDataTransform), typeof(ImageLoadingTransformerByte), null, typeof(SignatureLoadDataTransform),
   ImageLoadingTransformerByte.UserName, ImageLoadingTransformerByte.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageLoadingTransformerByte), null, typeof(SignatureLoadModel), "", ImageLoadingTransformerByte.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageLoadingTransformerByte), null, typeof(SignatureLoadRowMapper), "", ImageLoadingTransformerByte.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="ImageLoadingEstimator"/>.
    /// </summary>
    public sealed class ImageLoadingTransformerByte : OneToOneTransformerBase
    {
        internal sealed class Column : OneToOneColumn
        {
            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                return TryUnparseCore(sb);
            }
        }

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Multiple | ArgumentType.Required, HelpText = "New column definition(s) (optional form: name:src)", Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Folder where to search for images", ShortName = "folder")]
            public string ImageFolder;
        }

        internal const string Summary = "Load images from files to VBuffer<byte>.";
        internal const string UserName = "Image Loader Transform Byte";
        internal const string LoaderSignature = "ImageLoaderTransformByte";

        /// <summary>
        /// The folder to load the images from.
        /// </summary>
        public readonly string ImageFolder;

        /// <summary>
        /// The columns passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of <see cref="ImageLoadingTransformerByte"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingTransformerByte(IHostEnvironment env, string imageFolder = null, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingTransformerByte)), columns)
        {
            ImageFolder = imageFolder;
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView data)
        {
            return new ImageLoadingTransformerByte(env, options.ImageFolder, options.Columns.Select(x => (x.Name, x.Source ?? x.Name)).ToArray())
                .MakeDataTransform(data);
        }

        // Factory method for SignatureLoadModel.
        private static ImageLoadingTransformerByte Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel(GetVersionInfo());
            return new ImageLoadingTransformerByte(env.Register(nameof(ImageLoadingTransformerByte)), ctx);
        }

        private ImageLoadingTransformerByte(IHost host, ModelLoadContext ctx)
            : base(host, ctx)
        {
            // *** Binary format ***
            // <base>
            // int: id of image folder

            ImageFolder = ctx.LoadStringOrNull();
        }

        // Factory method for SignatureLoadDataTransform.
        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
            => Create(env, ctx).MakeDataTransform(input);

        // Factory method for SignatureLoadRowMapper.
        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        private protected override void CheckInputColumn(DataViewSchema inputSchema, int col, int srcCol)
        {
            if (!(inputSchema[srcCol].Type is TextDataViewType))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", ColumnPairs[col].inputColumnName, TextDataViewType.Instance.ToString(), inputSchema[srcCol].Type.ToString());
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            // int: id of image folder

            base.SaveColumns(ctx);
            ctx.SaveStringOrNull(ImageFolder);
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "IMGLOADR",
                //verWrittenCur: 0x00010001, // Initial
                verWrittenCur: 0x00010002, // Swith from OpenCV to Bitmap
                verReadableCur: 0x00010002,
                verWeCanReadBack: 0x00010002,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(ImageLoadingTransformer).Assembly.FullName);
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImageLoadingTransformerByte _parent;

            public Mapper(ImageLoadingTransformerByte parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
            }

            public static int LoadDataIntoBuffer(string path, ref VBuffer<byte> imgData)
            {
                //Console.WriteLine("Testing");
                int count = -1;
                // bufferSize == 1 used to avoid unnecessary buffer in FileStream
                using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1))
                {
                    long fileLength = fs.Length;
                    if (fileLength > int.MaxValue)
                        throw new IOException($"File {path} too big to open.");
                    else if (fileLength == 0)
                    {
                        byte[] imageBuffer;

                        // Some file systems (e.g. procfs on Linux) return 0 for length even when there's content.
                        // Thus we need to assume 0 doesn't mean empty.
                        imageBuffer = File.ReadAllBytes(path);
                        count = imageBuffer.Length;
                        Console.WriteLine("File length is zero");
                    }

                    count = (int)fileLength;
                    var editor = VBufferEditor.Create(ref imgData, count);
                    ReadToEnd(fs, editor.GetValues);
                    //var buffer = File.ReadAllBytes(path);
                    //fs.Read(editor.Values);
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

            public static byte[] ReadToEnd(System.IO.Stream stream, byte[] readBuffer)
            {
                long originalPosition = 0;

                if (stream.CanSeek)
                {
                    originalPosition = stream.Position;
                    stream.Position = 0;
                }

                try
                {

                    int totalBytesRead = 0;
                    int bytesRead;

                    while ((bytesRead = stream.Read(readBuffer, totalBytesRead, readBuffer.Length - totalBytesRead)) > 0)
                    {
                        totalBytesRead += bytesRead;

                        if (totalBytesRead == readBuffer.Length)
                        {
                            int nextByte = stream.ReadByte();
                            if (nextByte != -1)
                            {
                                byte[] temp = new byte[readBuffer.Length * 2];
                                Buffer.BlockCopy(readBuffer, 0, temp, 0, readBuffer.Length);
                                Buffer.SetByte(temp, totalBytesRead, (byte)nextByte);
                                readBuffer = temp;
                                totalBytesRead++;
                            }
                        }
                    }

                    return readBuffer;
                }
                finally
                {
                    if (stream.CanSeek)
                    {
                        stream.Position = originalPosition;
                    }
                }
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                disposer = null;
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                ReadOnlyMemory<char> src = default;
                ValueGetter<VBuffer<byte>> del =
                    (ref VBuffer<byte> dst) =>
                    {
                        //if (dst.Length > 0) // if not empty make it empty ?? Do we need to for VBuffer
                        //{
                        //    dst.;
                        //    dst = null;
                        //}

                        getSrc(ref src);

                        if (src.Length > 0)
                        {
                            string path = src.ToString();
                            if (!string.IsNullOrWhiteSpace(_parent.ImageFolder))
                                path = Path.Combine(_parent.ImageFolder, path);

                            int imgSize = LoadDataIntoBuffer(path, ref dst);
                            //dst = new Bitmap(path) { Tag = path };

                            // Check for an incorrect pixel format which indicates the loading failed
                            //if (dst.PixelFormat == System.Drawing.Imaging.PixelFormat.DontCare)
                            if(imgSize < 0)
                                throw Host.Except($"Failed to load image {src.ToString()}.");
                        }
                    };
                return del;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select(x => new DataViewSchema.DetachedColumn(x.outputColumnName, new VectorDataViewType(NumberDataViewType.Byte), null)).ToArray();
        }
    }

    /// <summary>
    /// <see cref="IEstimator{TTransformer}"/> for the <see cref="ImageLoadingTransformer"/>.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | [Text](<xref:Microsoft.ML.Data.TextDataViewType>) |
    /// | Output column data type | <xref:System.Drawing.Bitmap> |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.ImageAnalytics |
    ///
    /// The resulting <xref:Microsoft.ML.Data.ImageLoadingTransformer> creates a new column, named as specified in the output column name parameters, and
    /// loads in it images specified in the input column.
    /// Loading is the first step of almost every pipeline that does image processing, and further analysis on images.
    /// The images to load need to be in the formats supported by <xref:System.Drawing.Bitmap>.
    /// For end-to-end image processing pipelines, and scenarios in your applications, see the
    /// [examples](https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started) in the machinelearning-samples github repository.</a>
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="ImageEstimatorsCatalog.LoadImagesAsBytes(TransformsCatalog, string, string, string)" />

    public sealed class ImageLoadingEstimatorByte : TrivialEstimator<ImageLoadingTransformerByte>
    {

        /// <summary>
        /// Load images in memory.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingEstimatorByte(IHostEnvironment env, string imageFolder, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new ImageLoadingTransformerByte(env, imageFolder, columns))
        {
        }

        internal ImageLoadingEstimatorByte(IHostEnvironment env, ImageLoadingTransformerByte transformer)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingEstimator)), transformer)
        {
        }

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));
            var result = inputSchema.ToDictionary(x => x.Name);
            foreach (var (outputColumnName, inputColumnName) in Transformer.Columns)
            {
                if (!inputSchema.TryFindColumn(inputColumnName, out var col))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName);
                if (!(col.ItemType is TextDataViewType) || col.Kind != SchemaShape.Column.VectorKind.Scalar)
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", inputColumnName, TextDataViewType.Instance.ToString(), col.GetTypeString());

                result[outputColumnName] = new SchemaShape.Column(outputColumnName, SchemaShape.Column.VectorKind.Scalar, new VectorDataViewType(NumberDataViewType.Byte), false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
