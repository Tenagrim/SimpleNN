using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    [Serializable]
    class NN
    {
        public int[] Sizes { get { return sizes; } }
        public float[][] Layers { get { return layers; } }
        public float[][,] Weights { get { return weights; } }
        public float[] Output { get { return layers.Last(); } }
        public float[][] Errors { get { return errors_; } }

        private float[][] layers;
        private int[] sizes;
        private float[][,] weights;
        private float[][] errors_;
        static Random rand = new Random();
        private float l_c = 0.9F;

        public NN()
        {
            sizes = new int[] { 9, 10, 10, 4 };
            layers = new float[sizes.Length][];
            for (int i = 0; i < sizes.Length; i++)
            {
                layers[i] = new float[sizes[i]];
                layers[i][Sizes[i] - 1] = 1.0F;
            }
            errors_ = ErrorArr(layers);
            RandomFill(ref weights);
        }

        public NN(int[] sizes)
        {
            this.sizes = (int[])sizes.Clone();
            layers = new float[this.sizes.Length][];
            for (int i = 0; i < this.sizes.Length; i++)
            {
                if (i != sizes.Length - 1)
                {
                    layers[i] = new float[this.sizes[i] + 1];
                    layers[i][Sizes[i]] = 1.0F;
                    this.sizes[i]++;
                }
                else
                    layers[i] = new float[sizes[i]];
            }
             errors_ = ErrorArr(layers);
            RandomFill(ref weights);
        }
       
        public NN(NN reference)
        {
            sizes = reference.Sizes;

            layers = new float[sizes.Length][];
            for (int i = 0; i < sizes.Length; i++)
            {
                layers[i] = new float[sizes[i]];
                layers[i][Sizes[i] - 1] = 1.0F;
            }
            weights = (float[][,])reference.weights.Clone();
             errors_ = ErrorArr(layers);
            //Mutate();
        }

        public NN(NN[] parents)     // for genetic algorythms
        {
            sizes = parents[0].Sizes;

            layers = new float[sizes.Length][];
            for (int i = 0; i < sizes.Length; i++)
            {
                layers[i] = new float[sizes[i]];
                layers[i][Sizes[i] - 1] = 1.0F;
            }

            weights = new float[sizes.Length - 1][,];

            for (int i = 0; i < sizes.Length - 1; i++)
                weights[i] = new float[sizes[i], sizes[i + 1]];

            int rand_parent;
            for (int k = 0; k < weights.Length; k++)
            {
                for (int i = 0; i < weights[k].GetLength(0); i++)
                {
                    for (int j = 0; j < weights[k].GetLength(1); j++)
                    {
                        rand_parent = rand.Next(0, parents.Length - 1);
                        weights[k][i, j] = parents[rand_parent].weights[k][i, j];
                    }
                }
            }
             errors_ = ErrorArr(layers);
        }
        private void RandomFill(ref float[][,] weights)
        {
            weights = new float[sizes.Length - 1][,];

            for (int i = 0; i < sizes.Length - 1; i++)
                weights[i] = new float[sizes[i], sizes[i + 1]];

            float tmp;
            for (int k = 0; k < weights.Length; k++)
            {
                for (int i = 0; i < weights[k].GetLength(0); i++)
                {
                    for (int j = 0; j < weights[k].GetLength(1); j++)
                    {
                        tmp = (float)rand.NextDouble();
                        weights[k][i, j] = tmp * 2 - 1.0F;
                    }
                }
            }
        }
        public void Input(float[] data)
        {
            if (data.Length != layers[0].Length - 1)
                return;
            for (int i = 0; i < data.Length; i++)
                layers[0][i] = data[i];
        }

        public float[,] minus(float[,] a, float[,] b)
        {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
                return null;

            float[,] res = new float[a.GetLength(0), b.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                    res[i, j] = a[i, j] - b[i, j];
            }
            return res;
        }
        public float[] minus(float[] a, float[] b)
        {
            if (a.GetLength(0) != b.GetLength(0))
                return null;

            float[] res = new float[a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
                res[i] = a[i] - b[i];
            return res;
        }
        public float[,] transpose(float[] a)
        {
            float[,] res = new float[a.GetLength(0), 1];
            for (int i = 0; i < a.Length; i++)
                res[i, 0] = a[i];
            return res;
        }

        private float activation(float x)
        {
            return 1.0F / (1.0F + (float)Math.Exp(-x));     // sigmoid

            //return x > 0 ? 1 : 0;                         // relu
        }
        private float prev_sum(int i, int j)
        {
            float res = 0;
            for (int k = 0; k < sizes[i - 1]; k++)
            {
                res += layers[i - 1][k] * weights[i - 1][k, j];
            }
            return res;
        }
        public void Calc()
        {
            int len;
            for (int i = 1; i < layers.Length; i++)
            {
                len = i != layers.Length - 1 ? layers[i].Length - 1 : layers[i].Length;
                for (int j = 0; j < len; j++)
                {
                    layers[i][j] = activation(prev_sum(i, j));
                }
            }
        }

        public void Mutate()
        {
            float tmp = (float)rand.NextDouble();
            int i = rand.Next(0, Sizes.Length - 1);
            weights[i][rand.Next(0, weights[i].GetLength(0) - 1), rand.Next(0, weights[i].GetLength(1) - 1)] = tmp * 2 - 1.0F;
        }

        float F_relu(float x)
        {
            return (x < 0) ? 0 : 1;
        }

        private float[][] ErrorArr(float[][] ref_)
        {
            float[][] res = new float[ref_.Length - 1][];
            for (int i = 1; i < ref_.Length; i++)
                res[i - 1] = new float[ref_[i].Length];
            return res;
        }

        public void backpropagation(float[]input, float[] ref_values)
        {
            Input(input);
            Calc();

            List<float> errors;

            if (ref_values.Length != layers.Last().Length)
                return ;

            errors = new List<float>(layers.Last());
            
            //Вычисление ошибки выходного слоя
            for (int j = 0; j < layers.Last().Length; j++)
                errors_[errors_.Length-1][j] = ref_values[j] - layers.Last()[j];
            //Вычисление ошибки остальных слоев
            for (int i = errors_.Length -2; i >= 0; i--) 
            {
                for (int j = errors_[i].Length-1; j >= 0; j--)
                {
                    errors_[i][j] = 0;
                    for (int k = 0; k < errors_[i + 1].Length; k++)           //<---
                        errors_[i][j] += errors_[i + 1][k] * weights[i+1][j, k];
                }
            }


            //Корректировка весов всех слоев
            for (int k = 0; k < weights.Length; k++)
            {
                for (int i = 0; i < weights[k].GetLength(0); i++)
                    for (int j = 0; j < weights[k].GetLength(1); j++)
                        weights[k][i, j] += l_c * errors_[k][j] * (layers[k+1][j] * (1 - layers[k+1][j])) * layers[k][i];
            }
        
        }
    }
}
