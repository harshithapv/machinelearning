// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;

[assembly: LoadableClass(ImageLoadingTransformer.Summary, typeof(IDataTransform), typeof(ImageLoadingTransformer), typeof(ImageLoadingTransformer.Options), typeof(SignatureDataTransform),
    ImageLoadingTransformer.UserName, "ImageLoaderTransform", "ImageLoader")]

[assembly: LoadableClass(ImageLoadingTransformer.Summary, typeof(IDataTransform), typeof(ImageLoadingTransformer), null, typeof(SignatureLoadDataTransform),
   ImageLoadingTransformer.UserName, ImageLoadingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(ImageLoadingTransformer), null, typeof(SignatureLoadModel), "", ImageLoadingTransformer.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(ImageLoadingTransformer), null, typeof(SignatureLoadRowMapper), "", ImageLoadingTransformer.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="ImageLoadingEstimator"/>.
    /// </summary>
    public sealed class ImageLoadingTransformer : OneToOneTransformerBase
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

        internal const string Summary = "Load images from files.";
        internal const string UserName = "Image Loader Transform";
        internal const string LoaderSignature = "ImageLoaderTransform";

        /// <summary>
        /// The folder to load the images from.
        /// </summary>
        public readonly string ImageFolder;
        public readonly DataViewType Type;
        /// <summary>
        /// The columns passed to this <see cref="ITransformer"/>.
        /// </summary>
        internal IReadOnlyCollection<(string outputColumnName, string inputColumnName)> Columns => ColumnPairs.AsReadOnly();

        /// <summary>
        /// Initializes a new instance of <see cref="ImageLoadingTransformer"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingTransformer(IHostEnvironment env, string imageFolder = null, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingTransformer)), columns)
        {
            ImageFolder = imageFolder;
            Type = new ImageDataViewType();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="ImageLoadingTransformer"/>.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="type">DataView type - Image type or VBuffer of byte type</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingTransformer(IHostEnvironment env, string imageFolder = null , DataViewType type = null, params (string outputColumnName, string inputColumnName)[] columns)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingTransformer)), columns)
        {
            ImageFolder = imageFolder;
            if(type.Equals(null))
                Type = new ImageDataViewType();
            else
                Type = type;
        }

        // Factory method for SignatureDataTransform.
        internal static IDataTransform Create(IHostEnvironment env, Options options, IDataView data)
        {
            return new ImageLoadingTransformer(env, options.ImageFolder, null, options.Columns.Select(x => (x.Name, x.Source ?? x.Name)).ToArray())
                .MakeDataTransform(data);
        }

        // Factory method for SignatureLoadModel.
        private static ImageLoadingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel(GetVersionInfo());
            return new ImageLoadingTransformer(env.Register(nameof(ImageLoadingTransformer)), ctx);
        }

        private ImageLoadingTransformer(IHost host, ModelLoadContext ctx)
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

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema, Type);

        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImageLoadingTransformer _parent;
            //private readonly ImageDataViewType _imageType;
            private readonly DataViewType _type;

            public Mapper(ImageLoadingTransformer parent, DataViewSchema inputSchema, DataViewType type)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                //_imageType = new ImageDataViewType();
                _type = type;
                _parent = parent;
            }
            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                //var type = new VectorDataViewType(NumberDataViewType.Byte);
                disposer = null;
                if (new VectorDataViewType(NumberDataViewType.Byte).Equals(_type))
                {
                    return MakeGetterType(input, iinfo, activeOutput, (VectorDataViewType)_type, out disposer);
                }
                else
                {
                    return MakeGetterType(input, iinfo, activeOutput, (ImageDataViewType)_type, out disposer);
                }
            }

            public static int LoadDataIntoBuffer(string path, ref VBuffer<byte> imgData)
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
                        byte[] imageBuffer;

                        // Some file systems (e.g. procfs on Linux) return 0 for length even when there's content.
                        // Thus we need to assume 0 doesn't mean empty.
                        imageBuffer = File.ReadAllBytes(path);
                        count = imageBuffer.Length;
                        Console.WriteLine("File length is zero");
                    }

                    count = (int)fileLength;
                    var editor = VBufferEditor.Create(ref imgData, count);
                    //var buffer = File.ReadAllBytes(path);
#if NETSTANDARD2_1
                    fs.Read(editor.Values);
#else
                    int bytesread = ReadToEnd(fs, editor.GetValues);
#endif
                    imgData = editor.Commit();

                    return count;

                }

            }

            public static int ReadToEnd(System.IO.Stream stream, byte[] readBuffer)
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
                    return totalBytesRead;
                }
                finally
                {
                    if (stream.CanSeek)
                    {
                        stream.Position = originalPosition;
                    }
                }
            }

            private Delegate MakeGetterType(DataViewRow input, int iinfo, Func<int, bool> activeOutput, VectorDataViewType type, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                disposer = null;
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                ReadOnlyMemory<char> src = default;
                ValueGetter<VBuffer<byte>> del =
                    (ref VBuffer<byte> dst) =>
                    {
                        getSrc(ref src);

                        if (src.Length > 0)
                        {
                            string path = src.ToString();
                            if (!string.IsNullOrWhiteSpace(_parent.ImageFolder))
                                path = Path.Combine(_parent.ImageFolder, path);

                            int imgSize = LoadDataIntoBuffer(path, ref dst);
                            if (imgSize < 0)
                                throw Host.Except($"Failed to load image {src.ToString()}.");
                        }
                    };
                return del;
            }

            private Delegate MakeGetterType(DataViewRow input, int iinfo, Func<int, bool> activeOutput, ImageDataViewType type, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                disposer = null;
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                ReadOnlyMemory<char> src = default;
                ValueGetter<Bitmap> del =
                    (ref Bitmap dst) =>
                    {
                        if (dst != null)
                        {
                            dst.Dispose();
                            dst = null;
                        }

                        getSrc(ref src);

                        if (src.Length > 0)
                        {
                            string path = src.ToString();
                            if (!string.IsNullOrWhiteSpace(_parent.ImageFolder))
                                path = Path.Combine(_parent.ImageFolder, path);

                            dst = new Bitmap(path) { Tag = path };

                            // Check for an incorrect pixel format which indicates the loading failed
                            if (dst.PixelFormat == System.Drawing.Imaging.PixelFormat.DontCare)
                                throw Host.Except($"Failed to load image {src.ToString()}.");
                        }
                    };
                return del;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select(x => new DataViewSchema.DetachedColumn(x.outputColumnName, _type, null)).ToArray();
        }

        /*
        private sealed class Mapper : OneToOneMapperBase
        {
            private readonly ImageLoadingTransformer _parent;
            private readonly ImageDataViewType _imageType;

            public Mapper(ImageLoadingTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _imageType = new ImageDataViewType();
                _parent = parent;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
                Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);

                disposer = null;
                var getSrc = input.GetGetter<ReadOnlyMemory<char>>(input.Schema[ColMapNewToOld[iinfo]]);
                ReadOnlyMemory<char> src = default;
                ValueGetter<Bitmap> del =
                    (ref Bitmap dst) =>
                    {
                        if (dst != null)
                        {
                            dst.Dispose();
                            dst = null;
                        }

                        getSrc(ref src);

                        if (src.Length > 0)
                        {
                            string path = src.ToString();
                            if (!string.IsNullOrWhiteSpace(_parent.ImageFolder))
                                path = Path.Combine(_parent.ImageFolder, path);

                            dst = new Bitmap(path) { Tag = path };

                            // Check for an incorrect pixel format which indicates the loading failed
                            if (dst.PixelFormat == System.Drawing.Imaging.PixelFormat.DontCare)
                                throw Host.Except($"Failed to load image {src.ToString()}.");
                        }
                    };
                return del;
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
                => _parent.ColumnPairs.Select(x => new DataViewSchema.DetachedColumn(x.outputColumnName, _imageType, null)).ToArray();
        }
        */
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
    /// <seealso cref="ImageEstimatorsCatalog.LoadImages(TransformsCatalog, string, string, string)" />

    public sealed class ImageLoadingEstimator : TrivialEstimator<ImageLoadingTransformer>
    {
        private readonly DataViewType _imageType;

        /// <summary>
        /// Load images in memory.
        /// </summary>
        /// <param name="env">The host environment.</param>
        /// <param name="imageFolder">Folder where to look for images.</param>
        /// <param name="columns">Names of input and output columns.</param>
        internal ImageLoadingEstimator(IHostEnvironment env, string imageFolder, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new ImageLoadingTransformer(env, imageFolder, columns))
        {
        }

        internal ImageLoadingEstimator(IHostEnvironment env, string imageFolder, DataViewType type = null, params (string outputColumnName, string inputColumnName)[] columns)
            : this(env, new ImageLoadingTransformer(env, imageFolder, type, columns), type)
        {
        }

        internal ImageLoadingEstimator(IHostEnvironment env, ImageLoadingTransformer transformer, DataViewType type = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(ImageLoadingEstimator)), transformer)
        {
            if(type.Equals(null))
                _imageType = new ImageDataViewType();
            else
                _imageType = type;
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

                result[outputColumnName] = new SchemaShape.Column(outputColumnName, SchemaShape.Column.VectorKind.Scalar, _imageType, false);
            }

            return new SchemaShape(result.Values);
        }
    }
}
