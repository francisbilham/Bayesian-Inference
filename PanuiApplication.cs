using System;
using System.Collections.Generic;
using System.Text;

namespace Bayesian_Inference
{
    class PanuiApplication
    {
        protected string ID;
        protected List<Person> applicants;

        public PanuiApplication(string ID, List<Person> applicants = null)
        {
            this.ID = ID;
            this.applicants = applicants;

            this.applicants = applicants ?? new List<Person>();
        }

        public string getID()
        {
            return this.ID;
        }
        public List<Person> getApplicants()
        {
            return this.applicants;
        }
    }
}
