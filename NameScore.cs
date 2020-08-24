using System;
using System.Collections.Generic;
using System.Text;

namespace Bayesian_Inference
{
    class NameScore
    {
        protected string ID;
        protected double score;
        protected Name name1;
        protected Name name2;

        public NameScore(string ID, double score, Name name1, Name name2)
        {
            this.ID = ID;
            this.score = score;
            this.name1 = name1;
            this.name2 = name2;
        }

        public string getID()
        {
            return this.ID;
        }
        public double getScore()
        {
            return this.score;
        }

        public Name getName1()
        {
            return this.name1;
        }

        public Name getName2()
        {
            return this.name2;
        }
    }
}
