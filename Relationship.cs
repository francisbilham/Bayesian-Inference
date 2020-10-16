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
    class Relationship
    {
        protected Person person1;
        protected Person person2;
        protected List<Person> people;
        protected bool isPanui;
        protected bool isShareTrans;
        protected bool isNameMatch;
        protected double nameMatchScore; //
        protected double relatedScore; //
        protected int docCount; //fewest number of people appearing on the same common document
        protected double nameScore;
        protected bool isDeclared;
        protected Variable<bool> Panui;
        protected Variable<bool> ShareTrans;
        protected Variable<double> vNameScore;
        protected Variable<bool> PairScore;
        protected Variable<int> Related;

        public Relationship(Person person1, Person person2, List<Person> people = null, bool isPanui = false, bool isShareTrans = false, bool isNameMatch = false, double nameScore = double.NaN, bool isDeclared = false)
        {
            this.person1 = person1;
            this.person2 = person2;
            this.people = people ?? new List<Person>();

            if (this.people.Count == 0)
            {
                this.people.Add(person1);
                this.people.Add(person2);
            }

            this.isPanui = isPanui;
            this.isShareTrans = isShareTrans;
            this.isNameMatch = isNameMatch;
            this.nameScore = nameScore;
            this.isDeclared = isDeclared;
            this.Panui = Variable.New<bool>();
            this.ShareTrans = Variable.New<bool>();
            this.vNameScore = Variable.New<double>();
            this.PairScore = Variable.New<bool>();
            this.Related = Variable.New<int>().Named(this.person1.getName() + " + " + this.person2.getName());

            if (this.isPanui == false)
            {
                this.isPanui = findPanui(person1, person2, ref docCount);
            }

            if (this.isShareTrans == false)
            {
                this.isShareTrans = findShareTrans(person1, person2);
            }

            if (this.isNameMatch == false)
            {
                this.isNameMatch = findNameMatch(person1, person2);
            }

            if (double.IsNaN(this.nameScore))
            {
                this.nameScore = findNameMatchScore(person1, person2);
            }
        }

        public void setPanui(double bernoulli)
        {
            this.Panui.SetTo(Variable.Bernoulli(bernoulli));
        }
        public void setShareTrans(double bernoulli)
        {
            this.ShareTrans.SetTo(Variable.Bernoulli(bernoulli));
        }
        public void setvNameScore(double beta1, double beta2)
        {
            this.vNameScore.SetTo(Variable.Beta(beta1, beta2));
        }
        public void setPairScore(double bernoulli)
        {
            this.PairScore.SetTo(Variable.Bernoulli(bernoulli));
        }
        public void setRelated(double[] probs)
        {
            this.Related.SetTo(Variable.Discrete(probs));
        }

        public void observe()
        {
            this.Panui.ObservedValue = isPanui;
            this.ShareTrans.ObservedValue = isShareTrans;
            this.vNameScore.ObservedValue = nameScore;
        }

        public List<Person> getPeople()
        {
            return this.people;
        }

        public bool getIsPanui()
        {
            return this.isPanui;
        }
        public bool getIsShareTrans()
        {
            return this.isShareTrans;
        }
        public double getNameScore()
        {
            return this.nameScore;
        }
        public bool getIsDeclared()
        {
            return this.isDeclared;
        }
        public void declare()
        {
            this.isDeclared = true;
        }

        public Variable<bool> getPairScore()
        {
            return this.PairScore;
        }
        public Variable<int> getRelated()
        {
            return this.Related;
        }

        private bool findPanui(Person p1, Person p2, ref int personamount) //this doesnt find how many panui applications they have together, and only returns the amount of people in the first panui application it finds
        {
            bool ispanui = false; //initialise boolean

            for (int i = 0; i < p1.getApplications().Count; i++)
            {
                for (int j = 0; j < p2.getApplications().Count; j++)
                {
                    if (p1.getApplications()[i] == p2.getApplications()[j])
                    {
                        ispanui = true;

                        if (p1.getApplications()[i].getApplicants().Count < personamount)
                        {
                            personamount = p1.getApplications()[i].getApplicants().Count;
                        }
                    }
                }
            }
            return ispanui;
        }
        private bool findShareTrans(Person p1, Person p2)
        {
            for (int i = 0; i < p1.getShareTrans().Count; i++)
            {
                for (int j = 0; j < p2.getShareTrans().Count; j++)
                {
                    if (p1.getShareTrans()[i] == p2.getShareTrans()[j])
                    {
                        return true;
                    }
                }
            }
            return false;
        }
        private bool findNameMatch(Person p1, Person p2)
        {
            for (int i = 0; i < p1.getOtherNames().Count; i++)
            {
                for (int j = 0; j < p2.getOtherNames().Count; j++)
                {
                    if (p1.getOtherNames()[i] == p2.getOtherNames()[j])
                    {
                        return true;
                    }
                }
            }
            return false;
        }
        private double findNameMatchScore(Person p1, Person p2)
        {
            for (int i = 0; i < p1.getOtherNames().Count; i++) //loop through person 1's name nodes
            {
                for (int scores1 = 0; scores1 < p1.getOtherNames()[i].getNameScores().Count; scores1++) //loop through the list of name scores of each of the first persons name tags
                {
                    for (int j = 0; j < p2.getOtherNames().Count; j++) //loop through person 2's name nodes
                    {
                        for (int scores2 = 0; scores2 < p2.getOtherNames()[j].getNameScores().Count; scores2++) //loop through name scores of each of the second persons name tags
                        {
                            if (p1.getOtherNames()[i].getNameScores()[scores1].getID() == p2.getOtherNames()[j].getNameScores()[scores2].getID()) //if the people have name tags with a score between them
                            {
                                return p1.getOtherNames()[i].getNameScores()[scores1].getScore(); // return the score
                            }
                        }
                    }
                }

            }
            double score = CalculateSimilarity(p1.getName(), p2.getName());
            return score;
        }
        private static int CalcLevenshteinDistance(string a, string b) //social.technet.microsoft.com/wiki/contents/articles/26805.c-calculating-percentage-similarity-of-2-strings.aspx
        {
            if (String.IsNullOrEmpty(a) && String.IsNullOrEmpty(b))
            {
                return 0;
            }
            if (String.IsNullOrEmpty(a))
            {
                return b.Length;
            }
            if (String.IsNullOrEmpty(b))
            {
                return a.Length;
            }
            int lengthA = a.Length;
            int lengthB = b.Length;
            var distances = new int[lengthA + 1, lengthB + 1];
            for (int i = 0; i <= lengthA; distances[i, 0] = i++) ;
            for (int j = 0; j <= lengthB; distances[0, j] = j++) ;

            for (int i = 1; i <= lengthA; i++)
                for (int j = 1; j <= lengthB; j++)
                {
                    int cost = b[j - 1] == a[i - 1] ? 0 : 1;
                    distances[i, j] = Math.Min
                        (
                        Math.Min(distances[i - 1, j] + 1, distances[i, j - 1] + 1),
                        distances[i - 1, j - 1] + cost
                        );
                }
            return distances[lengthA, lengthB];
        }
        private static double CalculateSimilarity(string source, string target) //social.technet.microsoft.com/wiki/contents/articles/26805.c-calculating-percentage-similarity-of-2-strings.aspx
        {
            if ((source == null) || (target == null)) return 0.0;
            if ((source.Length == 0) || (target.Length == 0)) return 0.0;
            if (source == target) return 1.0;

            int stepsToSame = CalcLevenshteinDistance(source, target);
            return (1.0 - ((double)stepsToSame / (double)Math.Max(source.Length, target.Length)));
        }
    }
}