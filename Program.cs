﻿using System;
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
                Person JW = new Person("John Wilson");
                Person AJW = new Person("Adam John Wirihana");
                Person KWW = new Person("Koro Wipaaka Wilson");
                Person KWPW = new Person("Koro Wi Paaka Wilson");
                Person KDW = new Person("Kiri Dawn Wirihana");
                Person HTWTM = new Person("Horina Tutaki Wharemate Te Moa");
                Person FB = new Person("Francis Bilham");
                Person MS = new Person("Miyanda Siamoongwa");

                List<Person> personList = new List<Person>()
                {
                    JW,
                    AJW,
                    KWW,
                    KWPW,
                    KDW,
                    HTWTM,
                    FB,
                    MS
                };

                List<Person> p1 = new List<Person>();
                p1.Add(JW);
                p1.Add(KWW);
                List<Person> p2 = new List<Person>();
                p2.Add(JW);
                p2.Add(KWPW);
                List<Person> st1 = new List<Person>();
                st1.Add(AJW);
                st1.Add(HTWTM);
                List<Person> st2 = new List<Person>();
                st2.Add(KDW);
                st2.Add(HTWTM);
                List<Person> p3 = new List<Person>();
                p3.Add(FB);
                p3.Add(MS);

                List<PanuiApplication> applicationList = new List<PanuiApplication>()
                {
                    new PanuiApplication("1",p1),
                    new PanuiApplication("2",p2),
                    new PanuiApplication("3",p3)
                };

                List<ShareTransfer> transferList = new List<ShareTransfer>()
                {
                    new ShareTransfer("1",st1),
                    new ShareTransfer("2",st2)
                };

                List<Name> nameList = new List<Name>()
                {
                    new Name("John Wilson"),
                    new Name("Adam John Wirihana"),
                    new Name("Koro Wipaaka Wilson"),
                    new Name("Koro Wi Paaka Wilson"),
                    new Name("Kiri Dawn Wirihana"),
                    new Name("Horina Tutaki Wharemate Te Moa"),
                    new Name("Francis Bilham"),
                    new Name("Miyanda Siamoongwa")
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
                    for (int j = i + 1; j < personList.Count; j++)
                    {
                        relationshipList.Add(new Relationship(personList[i], personList[j]));
                    }
                }

                // Create triples and list them all in unsolved triple list
                List<Triple> tripleList = new List<Triple>();
                for (int i = 0; i < personList.Count - 2; i++)
                {
                    for (int j = i + 1; j < personList.Count - 1; j++)
                    {
                        if (personList[i] != personList[j])
                        {
                            for (int k = j + 1; k < personList.Count; k++)
                            {
                                if ((personList[i] != personList[k]) & (personList[j] != personList[k]))
                                {
                                    bool ij = isRelationship(personList[i], personList[j], relationshipList);
                                    bool ik = isRelationship(personList[i], personList[k], relationshipList);
                                    bool jk = isRelationship(personList[j], personList[k], relationshipList);
                                    if (ij & ik & jk)
                                    {
                                        List<Person> people = new List<Person>();
                                        people.Add(personList[i]);
                                        people.Add(personList[j]);
                                        people.Add(personList[k]);
                                        List<Relationship> pairs = new List<Relationship>();
                                        pairs.Add(GetRelationship(personList[i], personList[j], relationshipList));
                                        pairs.Add(GetRelationship(personList[i], personList[k], relationshipList));
                                        pairs.Add(GetRelationship(personList[j], personList[k], relationshipList));
                                        tripleList.Add(new Triple(pairs, people));
                                    }
                                }
                            }
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
                IDictionary<Triple, Variable<int>> Roots = new Dictionary<Triple, Variable<int>>(); //dictionary of the triple nodes.
                //IDictionary<string, Tuple<int, string, string, string>> RootInfo = new Dictionary<string, Tuple<int, string, string, string>>();

                double p = 0.05; // prior probability for any pair being related
                double q = 0.8; // setting to 1.0 means that 12 related and 13 related implies 23 related, setting to prior_prob_of_pair_related means that 12 / 13 related are independent of 23 related
                double z = 0.9; // prior probability of a pair being related (not the same person) given that they are related from triple analysis

                // a bunch of maths that relates to a 3 circle Venn diagram to compute the probability of the 8 outcomes for three relationships
                double D = p * p * q;
                double C = (p * p) * (1 - q);
                double B = p - ((2 - q) * (p * p));
                double A = 1 - (3 * p) + ((3 - q) * (p * p));

                double[] probs_noparent = new double[] { A, B, B, B, C, C, C, D };
                double[] probs_oneparent_set_0 = new double[] { A / (1 - p), B / (1 - p), B / (1 - p), C / (1 - p) };
                double[] probs_oneparent_set_1 = new double[] { B / p, C / p, C / p, D / p };
                double[] probs_twoparent_set_00 = new double[] { A / (A + B), B / (A + B) };
                double[] probs_twoparent_set_10_or_01 = new double[] { B / (B + C), C / (B + C) };
                double[] probs_twoparent_set_11 = new double[] { C / (C + D), D / (C + D) };

                // relationship outcomes depending on the triple analysis - related is now split into related and same person
                double[] probs_not_related = new double[] { 1.0, 0.0, 0.0 };
                double[] probs_related = new double[] { 0.0, z, (1 - z) };

                // Algorithm that sorts through triples, ranked by highest number of parents
                // pop triple once it has been solved
                int np;
                Triple current;
                List<Person> persons;
                while (tripleList.Count > 0)
                {
                    np = maxParents(tripleList);
                    current = tripleList[np];
                    persons = current.getPeople();
                    //Adds triple to triple dictionary
                    if (Roots.ContainsKey(current) == false)
                    {
                        Roots.Add(current, Variable.New<int>().Named(persons[0].getName() + " + " + persons[1].getName() + " + " + persons[2].getName()));
                    }


                    if (current.nParents() == 0)
                    {
                        if (Roots[current].IsDefined == false)
                        {
                            Roots[current].SetTo(Variable.Discrete(probs_noparent));
                        }

                        using (Variable.If(Roots[current] == 0))
                        {
                            current.getRelationships()[0].setPairScore(0);
                            current.getRelationships()[1].setPairScore(0);
                            current.getRelationships()[2].setPairScore(0);
                        }
                        using (Variable.If(Roots[current] == 1))
                        {
                            current.getRelationships()[0].setPairScore(1);
                            current.getRelationships()[1].setPairScore(0);
                            current.getRelationships()[2].setPairScore(0);
                        }
                        using (Variable.If(Roots[current] == 2))
                        {
                            current.getRelationships()[0].setPairScore(0);
                            current.getRelationships()[1].setPairScore(1);
                            current.getRelationships()[2].setPairScore(0);
                        }
                        using (Variable.If(Roots[current] == 3))
                        {
                            current.getRelationships()[0].setPairScore(0);
                            current.getRelationships()[1].setPairScore(0);
                            current.getRelationships()[2].setPairScore(1);
                        }
                        using (Variable.If(Roots[current] == 4))
                        {
                            current.getRelationships()[0].setPairScore(1);
                            current.getRelationships()[1].setPairScore(1);
                            current.getRelationships()[2].setPairScore(0);
                        }
                        using (Variable.If(Roots[current] == 5))
                        {
                            current.getRelationships()[0].setPairScore(1);
                            current.getRelationships()[1].setPairScore(0);
                            current.getRelationships()[2].setPairScore(1);
                        }
                        using (Variable.If(Roots[current] == 6))
                        {
                            current.getRelationships()[0].setPairScore(0);
                            current.getRelationships()[1].setPairScore(1);
                            current.getRelationships()[2].setPairScore(1);
                        }
                        using (Variable.If(Roots[current] == 7))
                        {
                            current.getRelationships()[0].setPairScore(1);
                            current.getRelationships()[1].setPairScore(1);
                            current.getRelationships()[2].setPairScore(1);
                        }

                    }
                    if (current.nParents() == 1)
                    {
                        List<Relationship> parents = current.getParents();
                        Relationship parent = parents[0];
                        using (Variable.IfNot(parent.getPairScore())) 
                        {
                            Roots[current].SetTo(Variable.Discrete(probs_oneparent_set_0));
                        }
                        using (Variable.If(parent.getPairScore())) 
                        {
                            Roots[current].SetTo(Variable.Discrete(probs_oneparent_set_1));
                        }

                        using (Variable.If(Roots[current] == 0))
                        {
                            current.getChildren()[0].setPairScore(0);
                            current.getChildren()[1].setPairScore(0);
                        }
                        using (Variable.If(Roots[current] == 1))
                        {
                            current.getChildren()[0].setPairScore(1);
                            current.getChildren()[1].setPairScore(0);
                        }
                        using (Variable.If(Roots[current] == 2))
                        {
                            current.getChildren()[0].setPairScore(0);
                            current.getChildren()[1].setPairScore(1);
                        }
                        using (Variable.If(Roots[current] == 3))
                        {
                            current.getChildren()[0].setPairScore(1);
                            current.getChildren()[1].setPairScore(1);
                        }
                    }
                    if (current.nParents() == 2)
                    {
                        List<Relationship> parents = current.getParents();
                        Relationship parent1 = parents[0];
                        Relationship parent2 = parents[1];
                        using (Variable.IfNot(parent1.getPairScore())) 
                        {
                            using (Variable.IfNot(parent2.getPairScore())) 
                            {
                                if (Roots[current].IsDefined == false)
                                {
                                    Roots[current].SetTo(Variable.Discrete(probs_twoparent_set_00));
                                }
                            }
                            using (Variable.If(parent2.getPairScore())) 
                            {
                                if (Roots[current].IsDefined == false)
                                {
                                    Roots[current].SetTo(Variable.Discrete(probs_twoparent_set_10_or_01));
                                }
                            }
                        }
                        using (Variable.If(parent1.getPairScore())) 
                        {
                            using (Variable.IfNot(parent2.getPairScore())) 
                            {
                                if (Roots[current].IsDefined == false)
                                {
                                    Roots[current].SetTo(Variable.Discrete(probs_twoparent_set_10_or_01));
                                }
                            }
                            using (Variable.If(parent2.getPairScore())) 
                            {
                                if (Roots[current].IsDefined == false)
                                {
                                    Roots[current].SetTo(Variable.Discrete(probs_twoparent_set_11));
                                }
                            }
                        }
                        using (Variable.If(Roots[current] == 0))
                        {
                            current.getChildren()[0].setPairScore(0);
                        }
                        using (Variable.If(Roots[current] == 1))
                        {
                            current.getChildren()[0].setPairScore(1);
                        }
                    }
                    current.declareChildren();
                    current.solve();
                    tripleList.Remove(current);
                }
                // loop over pairs and set panui, share transfer, namescore variables
                for (int i = 0; i < relationshipList.Count; i++)
                {
                    using (Variable.If(relationshipList[i].getPairScore()))
                    {
                        relationshipList[i].setRelated(probs_related);
                    }
                    using (Variable.IfNot(relationshipList[i].getPairScore()))
                    {
                        relationshipList[i].setRelated(probs_not_related);
                    }
                    using (Variable.If(relationshipList[i].getRelated() == 0)) // not related
                    {
                        relationshipList[i].setPanui(0.05);
                        relationshipList[i].setShareTrans(0.01);
                        relationshipList[i].setvNameScore(3.0, 20.0);
                    }
                    using (Variable.If(relationshipList[i].getRelated() == 1)) // related
                    {
                        relationshipList[i].setPanui(0.7);
                        relationshipList[i].setShareTrans(0.8);
                        relationshipList[i].setvNameScore(4.0, 8.0);
                    }
                    using (Variable.If(relationshipList[i].getRelated() == 2)) // same person
                    {
                        relationshipList[i].setPanui(0.0);
                        relationshipList[i].setShareTrans(0.0);
                        relationshipList[i].setvNameScore(16.0, 8.0);
                    }
                    relationshipList[i].observe();
                }

                InferenceEngine ie = new InferenceEngine();
                //ie.ShowFactorGraph = true;
                ie.Algorithm = new Microsoft.ML.Probabilistic.Algorithms.ExpectationPropagation();

                Console.WriteLine("First number: Probability they are not related");
                Console.WriteLine("Second number: Probability they are related");
                Console.WriteLine("Third number: Probability they are the same person");
                for (int i = 0; i < relationshipList.Count; i++)
                {
                    List<Person> people = relationshipList[i].getPeople();
                    Console.WriteLine(people[0].getName() + " and " + people[1].getName() + " are related " + ie.Infer(relationshipList[i].getRelated()));

                }

                //Console.WriteLine("breakpoint");
                //ie.ShowFactorGraph = true;

            }
            Run();
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

        // this function returns a relationship given 2 people, if the relationship exists in the network
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

        // this function takes 2 people and determines if a relationship between both of them exists in the network
        public static bool isRelationship(Person person1, Person person2, List<Relationship> relationshipList)
        {
            for (int i = 0; i < relationshipList.Count; i++)
            {
                if ((relationshipList[i].getPeople()[0] == person1) || (relationshipList[i].getPeople()[0] == person2))
                {
                    if ((relationshipList[i].getPeople()[1] == person1) || (relationshipList[i].getPeople()[1] == person2))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        // this function returns the index within the list of triples of the triple with the most parent relationships
        public static int maxParents(List<Triple> tripleList)
        {
            int n = -1;
            int index = 0;
            for (int i = 0; i < tripleList.Count; i++)
            {
                if (tripleList[i].nParents() > n)
                {
                    n = tripleList[i].nParents();
                    index = i;
                }
            }
            return index;
        }
    }
}