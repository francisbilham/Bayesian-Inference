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
                double total;
                double prior_prob_of_pair_related = 0.05; // prior probability for any pair being related
                double transfer_related_factor = 0.8; // setting to 1.0 means that 12 related and 13 related implies 23 related, 0.0 means that 12 / 13 related are independent of 23 related

                // a bunch of maths that relates to a 3 circle Venn diagram to compute the probability of the 8 outcomes for three relationships
                double prob_all_related = transfer_related_factor * prior_prob_of_pair_related * prior_prob_of_pair_related + (1 - transfer_related_factor) * prior_prob_of_pair_related * prior_prob_of_pair_related * prior_prob_of_pair_related;
                double prob_two_pairs_related = prior_prob_of_pair_related * prior_prob_of_pair_related - prob_all_related;
                double prob_one_pair_related = prior_prob_of_pair_related - prob_all_related - 2 * prob_two_pairs_related;
                double prob_none_related = 1 - 3 * prob_one_pair_related - 3 * prob_two_pairs_related - prob_all_related;
                
                double[] probs = new double[] { prob_none_related, prob_one_pair_related, prob_one_pair_related, prob_one_pair_related, prob_two_pairs_related, prob_two_pairs_related, prob_two_pairs_related, prob_all_related };

                // some hardcoded probability updates for 
                double[] probs_first_pair_set_0 = new double[] { prob_none_related, 0, prob_one_pair_related, prob_one_pair_related, 0, 0, prob_two_pairs_related, 0 };
                total = 0;
                for (int i = 0; i < 8; i++)
                {
                    total += probs_first_pair_set_0[i];
                }
                for (int i = 0; i < 8; i++)
                {
                    probs_first_pair_set_0[i] /= total;
                }

                double[] probs_first_pair_set_1 = new double[] { 0, prob_one_pair_related, 0, 0, prob_two_pairs_related, prob_two_pairs_related, 0, prob_all_related };
                total = 0;
                for (int i = 0; i < 8; i++)
                {
                    total += probs_first_pair_set_1[i];
                }
                for (int i = 0; i < 8; i++)
                {
                    probs_first_pair_set_1[i] /= total;
                }

                Variable<int> Root123 = Variable.Discrete(probs).Named("Root123"); // a node with 8 outcomes, 1 for each relationship outcome for three 'people'. e.g. if this is set to 7, then all three people are related.
                Variable<int> Root124 = Variable.New<int>().Named("Root124");

                Variable<bool> Relationship12 = Variable.New<bool>().Named("Relationship12");
                Variable<bool> Relationship13 = Variable.New<bool>().Named("Relationship13");
                Variable<bool> Relationship23 = Variable.New<bool>().Named("Relationship23");

                Variable<bool> Relationship14 = Variable.New<bool>().Named("Relationship14");
                Variable<bool> Relationship24 = Variable.New<bool>().Named("Relationship24");


                Variable<bool> Panui12 = Variable.New<bool>().Named("Panui12");
                Variable<bool> Panui13 = Variable.New<bool>().Named("Panui13");
                Variable<bool> Panui23 = Variable.New<bool>().Named("Panui23");
                
                Variable<bool> Panui14 = Variable.New<bool>().Named("Panui14");
                Variable<bool> Panui24 = Variable.New<bool>().Named("Panui24");

                using (Variable.If(Root123 == 0))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_0)); // if 12 are not related at the 123 root, then they mustn't be related at the 124 root
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 1))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_1)); // if 12 are related at the 123 root, then they must be related at the 124 root
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 2))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_0));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 3))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_0));
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root123 == 4))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_1));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 5))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_1));
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root123 == 6))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_0));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root123 == 7))
                {
                    Root124.SetTo(Variable.Discrete(probs_first_pair_set_1));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root124 == 0))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship14.SetTo(Variable.Bernoulli(0));
                    Relationship24.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root124 == 1))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship14.SetTo(Variable.Bernoulli(0));
                    Relationship24.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root124 == 2))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship14.SetTo(Variable.Bernoulli(1));
                    Relationship24.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root124 == 3))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship14.SetTo(Variable.Bernoulli(0));
                    Relationship24.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root124 == 4))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship14.SetTo(Variable.Bernoulli(1));
                    Relationship24.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root124 == 5))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship14.SetTo(Variable.Bernoulli(0));
                    Relationship24.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root124 == 6))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship14.SetTo(Variable.Bernoulli(1));
                    Relationship24.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root124 == 7))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship14.SetTo(Variable.Bernoulli(1));
                    Relationship24.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Relationship12))
                {
                    Panui12.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.IfNot(Relationship12))
                {
                    Panui12.SetTo(Variable.Bernoulli(0.005));
                }

                using (Variable.If(Relationship13))
                {
                    Panui13.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.IfNot(Relationship13))
                {
                    Panui13.SetTo(Variable.Bernoulli(0.005));
                }

                using (Variable.If(Relationship23))
                {
                    Panui23.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.IfNot(Relationship23))
                {
                    Panui23.SetTo(Variable.Bernoulli(0.005));
                }

                using (Variable.If(Relationship14))
                {
                    Panui14.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.IfNot(Relationship14))
                {
                    Panui14.SetTo(Variable.Bernoulli(0.005));
                }

                using (Variable.If(Relationship24))
                {
                    Panui24.SetTo(Variable.Bernoulli(0.7));
                }

                using (Variable.IfNot(Relationship24))
                {
                    Panui24.SetTo(Variable.Bernoulli(0.005));
                }

                InferenceEngine ie = new InferenceEngine();
                ie.Algorithm = new Microsoft.ML.Probabilistic.Algorithms.ExpectationPropagation();

                Panui12.ObservedValue = true;
                Panui13.ObservedValue = true;
                //Panui14.ObservedValue = true;
                //Panui24.ObservedValue = false;
                //Panui24.ObservedValue = false;
                //Relationship12.ObservedValue = false;
                //Relationship13.ObservedValue = true;
                Console.WriteLine(ie.Infer(Panui12));

                Console.WriteLine("\nRelationships\n");

                Console.Write("12: ");
                Console.WriteLine(ie.Infer(Relationship12));
                Console.Write("13: ");
                Console.WriteLine(ie.Infer(Relationship13));
                Console.Write("23: ");
                Console.WriteLine(ie.Infer(Relationship23));
                Console.Write("14: ");
                Console.WriteLine(ie.Infer(Relationship14));
                Console.Write("24: ");
                Console.WriteLine(ie.Infer(Relationship24));


            }
            Run();
        }

    }

}

