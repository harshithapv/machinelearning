
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

namespace Samples.Dynamic
{
    public class TestDecodeBMP
    {
        public static void Example()
        {
            string ImageFilePath = "C:\\Users\\havenka\\Desktop\\test\\tomato.jpg";

            try
            {
                byte[] b = File.ReadAllBytes(ImageFilePath);

                
                
                var bmp = new Bitmap(ImageFilePath);
                //var b2 = bmp.LockBits(...);

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }
        public static Bitmap ToImage(byte[] arr)
        {

            Bitmap bmp;
            using (var ms = new MemoryStream(arr))
            {
                bmp = new Bitmap(ms);
            }
            return bmp;

        }
    }
}

    


