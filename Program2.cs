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
                double prior_prob_of_pair_related = 0.05; // prior probability for any pair being related
                double transfer_related_factor = 0.8; // setting to 1.0 means that 12 related and 13 related implies 13 related, 0.0 means that 12 / 13 related are independent of 13 related

                // a bunch of maths that relates to a 3 circle Venn diagram to compute the probability of the 8 outcomes for three relationships
                double prob_all_related = transfer_related_factor * prior_prob_of_pair_related * prior_prob_of_pair_related + (1 - transfer_related_factor) * prior_prob_of_pair_related * prior_prob_of_pair_related * prior_prob_of_pair_related;
                double prob_two_pairs_related = prior_prob_of_pair_related * prior_prob_of_pair_related - prob_all_related;
                double prob_one_pair_related = prior_prob_of_pair_related - prob_all_related - 2 * prob_two_pairs_related;
                double prob_none_related = 1 - 3 * prob_one_pair_related - 3 * prob_two_pairs_related - prob_all_related;

                double[] probs = new double[] { prob_none_related, prob_one_pair_related, prob_one_pair_related, prob_one_pair_related, prob_two_pairs_related, prob_two_pairs_related, prob_two_pairs_related, prob_all_related };

                Variable<int> Root123 = Variable.Discrete(probs).Named("Root123");
                Variable<bool> Relationship12 = Variable.New<bool>().Named("Relationship12");
                Variable<bool> Relationship13 = Variable.New<bool>().Named("Relationship13");
                Variable<bool> Relationship23 = Variable.New<bool>().Named("Relationship23");

                Variable<bool> Panui12 = Variable.New<bool>().Named("Panui12");
                Variable<bool> Panui13 = Variable.New<bool>().Named("Panui13");
                Variable<bool> Panui23 = Variable.New<bool>().Named("Panui23");

                using (Variable.If(Root123 == 0))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 1))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 2))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 3))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root123 == 4))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(0));
                }

                using (Variable.If(Root123 == 5))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship13.SetTo(Variable.Bernoulli(0));
                    Relationship23.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root123 == 6))
                {
                    Relationship12.SetTo(Variable.Bernoulli(0));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(1));
                }

                using (Variable.If(Root123 == 7))
                {
                    Relationship12.SetTo(Variable.Bernoulli(1));
                    Relationship13.SetTo(Variable.Bernoulli(1));
                    Relationship23.SetTo(Variable.Bernoulli(1));
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


                InferenceEngine ie = new InferenceEngine();
                ie.Algorithm = new Microsoft.ML.Probabilistic.Algorithms.ExpectationPropagation();

                //prob_all_relatedPanui12.ObservedValue = false;
                Panui13.ObservedValue = true;
                //Panui23.ObservedValue = true;

                //Relationship12.ObservedValue = false;
                //Relationship13.ObservedValue = true;

                Console.WriteLine(ie.Infer(Relationship12));
                Console.WriteLine(ie.Infer(Relationship13));
                Console.WriteLine(ie.Infer(Relationship23));

            }
            Run();
        }

    }

}

