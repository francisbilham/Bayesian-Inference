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

                int n = personList.Count;
                Variable<bool>[,] Panui = new Variable<bool>[n, n];
                Variable<bool>[,] ShareTrans = new Variable<bool>[n, n];
                Variable<double>[,] NameScore = new Variable<double>[n, n];
                Variable<bool>[,] Related = new Variable<bool>[n, n];

                //List<List<Variable<bool>>> Panui = new List<List<Variable<bool>>>();
                //List<List<Variable<bool>>> ShareTrans = new List<List<Variable<bool>>>();
                //List<List<Variable<double>>> NameScore = new List<List<Variable<double>>>();
                //List<List<Variable<bool>>> Related = new List<List<Variable<bool>>>();

                // run inference on 3 person sub-networks
                for (int i = 0; i < personList.Count; i++)
                {
                    for (int j = 0; j < personList.Count; j++)
                    {
                        if (personList[i] != personList[j])
                        {
                            for (int k = 0; k < personList.Count; k++)
                            {
                                if ((personList[i] != personList[k]) & (personList[j] != personList[k]))
                                {
                                    string ij = i.ToString() + j.ToString();
                                    string ik = i.ToString() + k.ToString();
                                    string jk = j.ToString() + k.ToString();
                                    Panui[i,j] = Variable.New<bool>().Named("Panui" + ij);
                                    Panui[i,k] = Variable.New<bool>().Named("Panui" + ik);
                                    Panui[j,k] = Variable.New<bool>().Named("Panui" + jk);
                                    ShareTrans[i,j] = Variable.New<bool>().Named("ShareTrans" + ij);
                                    ShareTrans[i,k] = Variable.New<bool>().Named("ShareTrans" + ik);
                                    ShareTrans[j,k] = Variable.New<bool>().Named("ShareTrans" + jk);
                                    NameScore[i,j] = Variable.New<double>().Named("NameScore" + ij);
                                    NameScore[i,k] = Variable.New<double>().Named("NameScore" + ik);
                                    NameScore[j,k] = Variable.New<double>().Named("NameScore" + jk);
                                    Related[i,j] = Variable.Bernoulli(0.01).Named("Related" + ij);
                                    Related[i,k] = Variable.Bernoulli(0.01).Named("Related" + ik);
                                    Related[j,k] = Variable.Bernoulli(0.01).Named("Related" + jk);

                                    using (Variable.If(Related[i,j]))
                                    {
                                        using (Variable.If(Related[i,k]))
                                        {
                                            Related[j,k] = Variable.Bernoulli(0.8);
                                        }
                                        using (Variable.IfNot(Related[i,k]))
                                        {
                                            Related[j,k] = Variable.Bernoulli(0.01);
                                        }
                                        Panui[i,j].SetTo(Variable.Bernoulli(0.7));
                                        ShareTrans[i,j].SetTo(Variable.Bernoulli(0.8));
                                        NameScore[i,j].SetTo(Variable.Beta(5.0, 2.0));
                                    }

                                    using (Variable.IfNot(Related[i,j]))
                                    {
                                        using (Variable.If(Related[i,k]))
                                        {
                                            Related[j,k] = Variable.Bernoulli(0.01);
                                        }
                                        using (Variable.IfNot(Related[i,k]))
                                        {
                                            Related[j,k] = Variable.Bernoulli(0.01);
                                        }
                                        Panui[i,j].SetTo(Variable.Bernoulli(0.2));
                                        ShareTrans[i,j].SetTo(Variable.Bernoulli(0.01));
                                        NameScore[i,j].SetTo(Variable.Beta(2.0, 5.0));
                                    }
                                }
                            }
                        }
                    }
                }

                InferenceEngine ie = new InferenceEngine();

                // Run inference on every relationship
                for (int i = 0; i < personList.Count; i++)
                {
                    for (int j = i+1; j < personList.Count; j++)
                    {
                        if (personList[i] != personList[j])
                        {
                            Relationship relation = GetRelationship(personList[i], personList[j], relationshipList);
                            Panui[i,j].ObservedValue = relation.getIsPanui();
                            ShareTrans[i,j].ObservedValue = relation.getIsShareTrans();
                            NameScore[i,j].ObservedValue = relation.getNameScore();

                            Console.WriteLine(personList[i].getName() + " and " + personList[j].getName() + " are related: " + ie.Infer(Related[i,j]));
                        }
                    }
                }
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

