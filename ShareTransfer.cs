using System;
using System.Collections.Generic;
using System.Text;

namespace Bayesian_Inference
{
    class ShareTransfer
    {
        protected string ID;
        protected List<Person> shareholders;

        public ShareTransfer(string ID, List<Person> shareholders = null)
        {
            this.ID = ID;
            this.shareholders = shareholders;

            this.shareholders = shareholders ?? new List<Person>();
        }

        public string getID()
        {
            return this.ID;
        }
        public List<Person> getShareholders()
        {
            return this.shareholders;
        }
    }
}
