using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;





namespace Test
{
    class Program1
    {
        static void Main(string[] args)
        {

            void Run()
            {
                Variable<bool> Relationship12 = Variable.Bernoulli(0.001).Named("Relationship12");
                Variable<bool> Relationship13 = Variable.Bernoulli(0.001).Named("Relationship13");
                Variable<bool> Relationship23 = Variable.Bernoulli(0.001).Named("Relationship23");

                Variable<bool> Panui12 = Variable.New<bool>().Named("Panui12");
                Variable<bool> Panui13 = Variable.New<bool>().Named("Panui13");
                Variable<bool> Panui23 = Variable.New<bool>().Named("Panui23");

                //Relationship 2 3
                using (Variable.If(Relationship12))
                {
                    using (Variable.If(Relationship13))
                    {
                        Relationship23 = Variable.Bernoulli(1);
                    }
                    
                    using (Variable.IfNot(Relationship13))
                    {
                        Relationship23 = Variable.Bernoulli(0.01);
                    }
                }
                //Relationship 23
                using (Variable.IfNot(Relationship12))
                {
                    using (Variable.If(Relationship13))
                    {
                        Relationship23 = Variable.Bernoulli(0.01);
                    }

                    using (Variable.IfNot(Relationship13))
                    {
                        Relationship23 = Variable.Bernoulli(0.01);
                    }
                }



                //Relationship 1 2
                using (Variable.If(Relationship13))
                {
                    using (Variable.If(Relationship23))
                    {
                        Relationship12 = Variable.Bernoulli(1);
                    }

                    using (Variable.IfNot(Relationship23))
                    {
                        Relationship12 = Variable.Bernoulli(0.01);
                    }
                }

                //Relationship 1 2
                using (Variable.IfNot(Relationship13))
                {
                    using (Variable.If(Relationship23))
                    {
                        Relationship12 = Variable.Bernoulli(0.01);
                    }

                    using (Variable.IfNot(Relationship23))
                    {
                        Relationship12 = Variable.Bernoulli(0.01);
                    }
                }



                //Relationship 1 3
                using (Variable.If(Relationship12))
                {
                    using (Variable.If(Relationship23))
                    {
                        Relationship13 = Variable.Bernoulli(1);
                    }

                    using (Variable.IfNot(Relationship23))
                    {
                        Relationship13 = Variable.Bernoulli(0.01);
                    }
                }
                //Relationship 1 3
                using (Variable.IfNot(Relationship12))
                {
                    using (Variable.If(Relationship23))
                    {
                        Relationship13 = Variable.Bernoulli(0.01);
                    }

                    using (Variable.IfNot(Relationship23))
                    {
                        Relationship13 = Variable.Bernoulli(0.01);
                    }
                }


                using (Variable.If(Relationship12))
                {
                    Panui12.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.If(Relationship13))
                {
                    Panui13.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.If(Relationship23))
                {
                    Panui23.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.IfNot(Relationship12))
                {
                    Panui12.SetTo(Variable.Bernoulli(0.015));
                }

                using (Variable.IfNot(Relationship13))
                {
                    Panui13.SetTo(Variable.Bernoulli(0.015));
                }

                using (Variable.IfNot(Relationship23))
                {
                    Panui23.SetTo(Variable.Bernoulli(0.015));
                }






                //using (Variable.If(Relationship13))
                //{
                //    using (Variable.If(Relationship23))
                //    {
                //        Relationship12 = Variable.Bernoulli(1).Named("Relationship123");
                //    }
                //}

                InferenceEngine ie = new InferenceEngine();

                Panui12.ObservedValue = true;
                Panui13.ObservedValue = false;
                Panui23.ObservedValue = true;

                Relationship12.ObservedValue = true;
                Relationship13.ObservedValue = true;

                Console.WriteLine(ie.Infer(Relationship12));
                Console.WriteLine(ie.Infer(Relationship13));
                Console.WriteLine(ie.Infer(Relationship23));

                Console.WriteLine("jeff");



            }
            Run();
        }

    }

}

