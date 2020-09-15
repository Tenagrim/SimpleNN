using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SimpleNN
{
    public partial class Form1 : Form
    {
        private List<Point> points;
        private Graphics graphics;
        private NN nn;
        private Pen pen;
        private Brush brush;
        private float circle_size = 10;
        private int[] sizes = new int[] { 2, 1 };
        public Form1()
        {
            InitializeComponent();
            points = new List<Point>();
            pictureBox1.Image = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            graphics = Graphics.FromImage(pictureBox1.Image);
            //nn = new NN(new int[] { 2, 2 });
            nn = new NN(sizes);
            pen = new Pen(Color.White, 2.0F);
            //brush = Brushes.
        }

        private void DisplayPoints()
        {
            foreach (var p in points)
            {
                if (p.state == 0)
                {
                    brush = Brushes.ForestGreen;
                    //pen.Color = Color.FromArgb(0, 255, 0);
                }
                else if (p.state == 1)
                {
                    //pen.Color = Color.FromArgb(0, 0, 255);
                    brush = Brushes.Blue;
                }
                graphics.DrawEllipse(pen, p.x - circle_size / 2, p.y - circle_size / 2, circle_size, circle_size);
                graphics.FillEllipse(brush, p.x - circle_size / 2, p.y - circle_size / 2, circle_size, circle_size);
            }
        }

        private void DisplayWeights()
        {
            richTextBox1.Text = "";
            for (int i = 0; i < nn.Weights.Length; i++)
            {
                for (int j = 0; j < nn.Weights[i].GetLength(0); j++)
                {
                    for (int k = 0; k < nn.Weights[i].GetLength(1); k++)
                        richTextBox1.Text += nn.Weights[i][j, k] + "\t";
                    richTextBox1.Text += "\n";
                }
                richTextBox1.Text += "=========";
            }
        }

        private void Display()
        {
            Color col;

            float[] input = new float[2];
            for (int i = 0; i < pictureBox1.Width; i++)
            {
                for (int j = 0; j < pictureBox1.Height; j++)
                {
                    input[0] = (float)i / pictureBox1.Width;
                    input[1] = (float)j / pictureBox1.Height;
                    nn.Input(input);
                    nn.Calc();
                    col = nn.Output[0] < 0.5F ? Color.FromArgb(0, 255, 0) : Color.FromArgb(0, 0, 255);
                    ((Bitmap)pictureBox1.Image).SetPixel(i, j, col); // Color.FromArgb(0, (int)((nn.Output[0] + 1) / 2 * 255), (int)((nn.Output[1] + 1) / 2 * 255)));
                }
            }
            DisplayPoints();
            DisplayWeights();
            pictureBox1.Refresh();
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
                points.Add(new Point(e.X, e.Y, 0));
            if (e.Button == MouseButtons.Right)
                points.Add(new Point(e.X, e.Y, 1));
            Display();
        }

        private void Form1_Shown(object sender, EventArgs e)
        {
            Display();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            float[] ref_values = new float[1];
            float[] input = new float[2];
            for (int j = 0; j < 300; j++)
                for (int i = 0; i < points.Count; i++)
                {
                    // ref_values[0] = points[i].state == 0 ? 1 : 0;
                    // ref_values[1] = points[i].state == 0 ? 0 : 1;
                    ref_values[0] = (float)points[i].state;
                    input[0] = (float)points[i].x / pictureBox1.Width;
                    input[1] = (float)points[i].y / pictureBox1.Height;
                    nn.backpropagation(input, ref_values);
                }
            Display();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            points.Clear();
            Display();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            nn = new NN(sizes);
            points.Clear();
            Display();
        }
    }
}
