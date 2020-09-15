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

namespace Bayesian_Inference
{
    class Program
    {
        static void Main(string[] args)
        {
            void Run()
            {
                /////////////////////////////////////////// LOADING IN NETWORK DATA ///////////////////////////////
                Person JR = new Person("Joe Rokocoko");
                Person JsR = new Person("Josevata Rokocoko");
                Person SS = new Person("Sitiveni Sivivatu");
                Person JV = new Person("Joeli Vidiri");

                List<Person> personList = new List<Person>()
                {
                    JR,
                    JsR,
                    SS,
                    JV
                };

                List<Person> p1 = new List<Person>();
                p1.Add(JR);
                p1.Add(SS);
                List<Person> p2 = new List<Person>();
                p2.Add(JR);
                p2.Add(JV);
                List<Person> st1 = new List<Person>();
                st1.Add(JsR);
                st1.Add(SS);
                List<Person> st2 = new List<Person>();
                st2.Add(JsR);
                st2.Add(JV);

                List<PanuiApplication> applicationList = new List<PanuiApplication>()
                {
                    new PanuiApplication("1",p1),
                    new PanuiApplication("2",p2)
                };

                List<ShareTransfer> transferList = new List<ShareTransfer>()
                {
                    new ShareTransfer("1",st1),
                    new ShareTransfer("2",st2)
                };

                List<Name> nameList = new List<Name>()
                {
                    new Name("Joe Rokocoko"),
                    new Name("Josevata Rokocoko"),
                    new Name("Sitiveni Sivivatu"),
                    new Name("Joeli Vidiri")
                };

                // add Panui Applications to each person
                for (int i = 0; i < applicationList.Count; i++)
                {
                    for (int j = 0; j < applicationList[i].getApplicants().Count; j++)
                    {
                        applicationList[i].getApplicants()[j].addPanui(applicationList[i]);
                    }
                }

                // add Share Transfers to each person
                for (int i = 0; i < transferList.Count; i++)
                {
                    for (int j = 0; j < transferList[i].getShareholders().Count; j++)
                    {
                        transferList[i].getShareholders()[j].addShareTransfer(transferList[i]);
                    }
                }

                // create relationships between every pair of people
                List<Relationship> relationshipList = new List<Relationship>();
                for (int i = 0; i < personList.Count; i++)
                {
                    for (int j = 0; j < personList.Count; j++)
                    {
                        if (personList[i] != personList[j])
                        {
                            relationshipList.Add(new Relationship(personList[i], personList[j]));

                        }
                    }
                }
                ///////////////////////////////// FINISHED LOADING IN NETWORK DATA //////////////////////////////////

                // Setting up arrays and dictionary. Each array is n by n (n being the amount of people in the network) and the elements correspond to 
                // a relationship ([1,2] is relates to relationship12) and so each element stores the status of the attribute in the corresponding relationship
                // e.g Panui[1,2] = true means people 1 and 2 appear in a panui application together.
                int n = personList.Count;
                Variable<bool>[,] Panui = new Variable<bool>[n, n];
                Variable<bool>[,] ShareTrans = new Variable<bool>[n, n];
                Variable<double>[,] NameScore = new Variable<double>[n, n];
                Variable<bool>[,] Related = new Variable<bool>[n, n];
                IDictionary<string, Variable<int>> Roots = new Dictionary<string, Variable<int>>(); //dictionary of the triple nodes.

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

                // run inference on 3 person sub-networks. Loops over i from 0 to n, then j from i + 1 to n, then k from j + 1 to n. Since the matrices we 
                // store data in are symmetrical, we can do this as long as we make sure all the operations for ij are applied to ji too.
                for (int i = 0; i < personList.Count; i++)
                {
                    for (int j = i + 1; j < personList.Count; j++)
                    {
                        if (personList[i] != personList[j])
                        {
                            for (int k = j + 1; k < personList.Count; k++)
                            {
                                if ((personList[i] != personList[k]) & (personList[j] != personList[k]))
                                {
                                    // strings of the indices
                                    string ij = i.ToString() + j.ToString();
                                    string ik = i.ToString() + k.ToString();
                                    string jk = j.ToString() + k.ToString();
                                    string ijk = i.ToString() + j.ToString() + k.ToString();

                                    // if the triple is not already linked in the network, then create it
                                    if (Roots.ContainsKey(ijk) == false)
                                    {
                                        Roots.Add(ijk, Variable.Discrete(probs).Named("Root" + ijk));
                                    }

                                    // initialising variables in related array 
                                    Related[i, j] = Variable.New<bool>();
                                    Related[i, k] = Variable.New<bool>();
                                    Related[j, k] = Variable.New<bool>();
                                    Related[j, i] = Variable.New<bool>();
                                    Related[k, i] = Variable.New<bool>();
                                    Related[k, j] = Variable.New<bool>();

                                    // initialising variables in panui array 
                                    Panui[i, j] = Variable.New<bool>();
                                    Panui[i, k] = Variable.New<bool>();
                                    Panui[j, k] = Variable.New<bool>();
                                    Panui[j, i] = Variable.New<bool>();
                                    Panui[k, i] = Variable.New<bool>();
                                    Panui[k, j] = Variable.New<bool>();

                                    // initialising variables in share transfer array
                                    ShareTrans[i, j] = Variable.New<bool>();
                                    ShareTrans[i, k] = Variable.New<bool>();
                                    ShareTrans[j, k] = Variable.New<bool>();
                                    ShareTrans[j, i] = Variable.New<bool>();
                                    ShareTrans[k, i] = Variable.New<bool>();
                                    ShareTrans[k, j] = Variable.New<bool>();

                                    // initialising variables in name score array
                                    NameScore[i, j] = Variable.New<double>();
                                    NameScore[i, k] = Variable.New<double>();
                                    NameScore[j, k] = Variable.New<double>();
                                    NameScore[j, i] = Variable.New<double>();
                                    NameScore[k, i] = Variable.New<double>();
                                    NameScore[k, j] = Variable.New<double>();

                                    //for loop which goes over adjacent triples and sets distribution based on result of this root triple
                                    for (int l = k + 1; l < personList.Count; l++)
                                    {
                                        //strings of all adjacent triples
                                        string ijl = ij + l.ToString();
                                        string ikl = ik + l.ToString();
                                        string jkl = jk + l.ToString();

                                        //create all triples that don't already exist in the bayesian network
                                        if (Roots.ContainsKey(ijl) == false)
                                        {
                                            Roots.Add(ijl, Variable.New<int>());
                                        }

                                        if (Roots.ContainsKey(ikl) == false)
                                        {
                                            Roots.Add(ikl, Variable.New<int>());
                                        }

                                        if (Roots.ContainsKey(jkl) == false)
                                        {
                                            Roots.Add(jkl, Variable.New<int>());
                                        }
                                    }

                                    using (Variable.If(Roots[ijk] == 0))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();

                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false) 
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(0));
                                        Related[i, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, j].SetTo(Variable.Bernoulli(0));
                                    }

                                    using (Variable.If(Roots[ijk] == 1))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();

                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false)
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(1));
                                        Related[i, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, j].SetTo(Variable.Bernoulli(0));
                                    }

                                    using (Variable.If(Roots[ijk] == 2))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();

                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false) 
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(0));
                                        Related[i, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, j].SetTo(Variable.Bernoulli(0));
                                    }

                                    using (Variable.If(Roots[ijk] == 3))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();
                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false)
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(0));
                                        Related[i, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, j].SetTo(Variable.Bernoulli(1));
                                    }

                                    using (Variable.If(Roots[ijk] == 4))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();

                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false)
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(1));
                                        Related[i, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, j].SetTo(Variable.Bernoulli(0));
                                    }

                                    using (Variable.If(Roots[ijk] == 5))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();

                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false)
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(1));
                                        Related[i, k].SetTo(Variable.Bernoulli(0));
                                        Related[j, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, j].SetTo(Variable.Bernoulli(1));
                                    }

                                    using (Variable.If(Roots[ijk] == 6))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();

                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false)
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_0).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(0));
                                        Related[i, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, i].SetTo(Variable.Bernoulli(0));
                                        Related[k, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, j].SetTo(Variable.Bernoulli(1));
                                    }

                                    using (Variable.If(Roots[ijk] == 7))
                                    {
                                        //for loop which goes over adjacent nodes and sets distribution based on result of this root node
                                        for (int l = k + 1; l < personList.Count; l++)
                                        {
                                            string ijl = ij + l.ToString();
                                            string ikl = ik + l.ToString();
                                            string jkl = jk + l.ToString();

                                            // if the adjacent triples haven't already had their distribution set previously, then set it based on this root triples results
                                            if (Roots[ijl].IsDefined == false)
                                            {
                                                Roots[ijl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ijl));
                                                Roots[ijl].Name = ("Root" + ijl);
                                            }
                                            if (Roots[ikl].IsDefined == false)
                                            {
                                                Roots[ikl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + ikl));
                                                Roots[ikl].Name = ("Root" + ikl);
                                            }
                                            if (Roots[jkl].IsDefined == false)
                                            {
                                                Roots[jkl].SetTo(Variable.Discrete(probs_first_pair_set_1).Named("Root" + jkl));
                                                Roots[jkl].Name = ("Root" + jkl);
                                            }
                                        }

                                        // set variables bernoulli distribution based on this triples outcome
                                        Related[i, j].SetTo(Variable.Bernoulli(1));
                                        Related[i, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, k].SetTo(Variable.Bernoulli(1));
                                        Related[j, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, i].SetTo(Variable.Bernoulli(1));
                                        Related[k, j].SetTo(Variable.Bernoulli(1));
                                    }


                                    using (Variable.If(Related[i, j]))
                                    {
                                        Panui[i, j].SetTo(Variable.Bernoulli(0.7));
                                        ShareTrans[i, j].SetTo(Variable.Bernoulli(0.8));
                                        NameScore[i, j].SetTo(Variable.Beta(2.0, 5.0));
                                        Panui[j, i].SetTo(Variable.Bernoulli(0.7));
                                        ShareTrans[j, i].SetTo(Variable.Bernoulli(0.8));
                                        NameScore[j, i].SetTo(Variable.Beta(2.0, 5.0));
                                    }
                                    using (Variable.IfNot(Related[i, j]))
                                    {
                                        Panui[i, j].SetTo(Variable.Bernoulli(0.2));
                                        ShareTrans[i, j].SetTo(Variable.Bernoulli(0.01));
                                        NameScore[i, j].SetTo(Variable.Beta(5.0, 2.0));
                                        Panui[j, i].SetTo(Variable.Bernoulli(0.2));
                                        ShareTrans[j, i].SetTo(Variable.Bernoulli(0.01));
                                        NameScore[j, i].SetTo(Variable.Beta(5.0, 2.0));
                                    }

                                    using (Variable.If(Related[i, k]))
                                    {
                                        Panui[i, k].SetTo(Variable.Bernoulli(0.7));
                                        ShareTrans[i, k].SetTo(Variable.Bernoulli(0.8));
                                        NameScore[i, k].SetTo(Variable.Beta(2.0, 5.0));
                                        Panui[k, i].SetTo(Variable.Bernoulli(0.7));
                                        ShareTrans[k, i].SetTo(Variable.Bernoulli(0.8));
                                        NameScore[k, i].SetTo(Variable.Beta(2.0, 5.0));
                                    }
                                    using (Variable.IfNot(Related[i, k]))
                                    {
                                        Panui[i, k].SetTo(Variable.Bernoulli(0.2));
                                        ShareTrans[i, k].SetTo(Variable.Bernoulli(0.01));
                                        NameScore[i, k].SetTo(Variable.Beta(5.0, 2.0));
                                        Panui[k, i].SetTo(Variable.Bernoulli(0.2));
                                        ShareTrans[k, i].SetTo(Variable.Bernoulli(0.01));
                                        NameScore[k, i].SetTo(Variable.Beta(5.0, 2.0));
                                    }

                                    using (Variable.If(Related[j, k]))
                                    {
                                        Panui[j, k].SetTo(Variable.Bernoulli(0.7));
                                        ShareTrans[j, k].SetTo(Variable.Bernoulli(0.8));
                                        NameScore[j, k].SetTo(Variable.Beta(2.0, 5.0));
                                        Panui[k, j].SetTo(Variable.Bernoulli(0.7));
                                        ShareTrans[k, j].SetTo(Variable.Bernoulli(0.8));
                                        NameScore[k, j].SetTo(Variable.Beta(2.0, 5.0));
                                    }
                                    using (Variable.IfNot(Related[j, k]))
                                    {
                                        Panui[j, k].SetTo(Variable.Bernoulli(0.2));
                                        ShareTrans[j, k].SetTo(Variable.Bernoulli(0.01));
                                        NameScore[j, k].SetTo(Variable.Beta(5.0, 2.0));
                                        Panui[k, j].SetTo(Variable.Bernoulli(0.2));
                                        ShareTrans[k, j].SetTo(Variable.Bernoulli(0.01));
                                        NameScore[k, j].SetTo(Variable.Beta(5.0, 2.0));
                                    }


                                }
                            }
                        }
                    }
                }

                InferenceEngine ie = new InferenceEngine();
                ie.Algorithm = new Microsoft.ML.Probabilistic.Algorithms.ExpectationPropagation();

                // Run inference on every relationship
                for (int i = 0; i < personList.Count; i++)
                {
                    for (int j = i + 1; j < personList.Count; j++)
                    {
                        if (personList[i] != personList[j])
                        {
                            Relationship relation = GetRelationship(personList[i], personList[j], relationshipList);
                            Panui[i, j].ObservedValue = relation.getIsPanui();
                            ShareTrans[i, j].ObservedValue = relation.getIsShareTrans();
                            NameScore[i, j].ObservedValue = relation.getNameScore();

                            //Console.WriteLine(personList[i].getName() + " and " + personList[j].getName() + " are related: " + ie.Infer(Related[i, j]));
                            
                        }
                    }
                }

                for (int i = 0; i < personList.Count; i++)
                {
                    for (int j = i + 1; j < personList.Count; j++)
                    {
                        if (personList[i] != personList[j])
                        {
                            Console.WriteLine(personList[i].getName() + " and " + personList[j].getName() + " are related: " + ie.Infer(Related[i, j]));
                        }
                    }
                }
            }
            Run();
            Console.WriteLine("breakpoint");
        }

        public static Person GetPerson(string name, List<Person> persons)
        {

            //Loop over the list of names and check if the name matches the person nodes display name or any other names linked to it. If so, return the node
            for (int i = 0; i < persons.Count; i++)
            {
                if (name == persons[i].getName())
                {
                    return persons[i];
                }
                else if (persons[i].getOtherNames() != null)
                {
                    for (int j = 0; j < persons[i].getOtherNames().Count; j++)
                    {
                        if (name == persons[i].getOtherNames()[j].getName())
                        {
                            return persons[i];
                        }
                    }
                }


            }
            return null; //if no person is found with that name then return nothing
        }

        public static Name GetName(string name, List<Name> names)
        {
            for (int i = 0; i < names.Count; i++)
            {
                if (name == names[i].getName())
                {
                    return names[i];
                }
            }
            return null;
        }

        public static ShareTransfer GetShareTransfer(string id, List<ShareTransfer> transfers)
        {
            for (int i = 0; i < transfers.Count; i++)
            {
                if (id == transfers[i].getID())
                {
                    return transfers[i];
                }
            }
            return null;
        }

        public static PanuiApplication GetApplication(string id, List<PanuiApplication> applications)
        {
            for (int i = 0; i < applications.Count; i++)
            {
                if (id == applications[i].getID())
                {
                    return applications[i];
                }
            }
            return null;
        }

        public static NameScore GetNameScore(string id, List<NameScore> namescores)
        {
            for (int i = 0; i < namescores.Count; i++)
            {
                if (id == namescores[i].getID())
                {
                    return namescores[i];
                }
            }
            return null;
        }

        public static Relationship GetRelationship(Person person1, Person person2, List<Relationship> relationshipList)
        {
            for (int i = 0; i < relationshipList.Count; i++)
            {
                if ((relationshipList[i].getPeople()[0] == person1) || (relationshipList[i].getPeople()[0] == person2))
                {
                    if ((relationshipList[i].getPeople()[1] == person1) || (relationshipList[i].getPeople()[1] == person2))
                    {
                        return relationshipList[i];
                    }
                }
            }
            return null;
        }

    }
}
