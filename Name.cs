using System;
using System.Collections.Generic;
using System.Text;

namespace Bayesian_Inference
{
    class Name
    {
        protected string name;
        protected List<Person> persons;
        protected List<NameScore> namescores;

        public Name(string name, List<Person> persons = null, List<NameScore> namescores = null)
        { 
            this.name = name;
            this.persons = persons;
            this.namescores = namescores;
            this.persons = persons ?? new List<Person>();
            this.namescores = namescores ?? new List<NameScore>();
        }

        public string getName()
        {
            return this.name;
        }

        public List<Person> getPersons()
        {
            return this.persons;
        }

        public List<NameScore> getNameScores()
        {
            return this.namescores;
        }
    }
}
