using System;
using System.Collections.Generic;
using System.Text;

namespace Bayesian_Inference
{
    class Person
    {
        protected string displayName;
        protected List<Person> relationships;
        protected List<Name> names;
        protected List<PanuiApplication> panuiApplications;
        protected List<ShareTransfer> shareTransfers;

        public Person(string displayName, List<Person> relationships = null, List<Name> names = null, List<ShareTransfer> shareTransfers = null, List<PanuiApplication> panuiApplications = null)
        {
            this.displayName = displayName;
            this.relationships = relationships;
            this.names = names;
            this.panuiApplications = panuiApplications;
            this.shareTransfers = shareTransfers;

            this.relationships = relationships ?? new List<Person>();
            this.names = names ?? new List<Name>();
            this.panuiApplications = panuiApplications ?? new List<PanuiApplication>();
            this.shareTransfers = shareTransfers ?? new List<ShareTransfer>();
        }

        public void addName(Name name)
        {
            this.names.Add(name);
        }
        public void addPanui(PanuiApplication panui)
        {
            this.panuiApplications.Add(panui);
        }
        public void addShareTransfer(ShareTransfer st)
        {
            this.shareTransfers.Add(st);
        }
        public void addPerson(Person person)
        {
            this.relationships.Add(person);
        }

        public string getName()
        {
            return this.displayName;
        }
        public List<Name> getOtherNames()
        {
            return this.names;
        }
        public List<PanuiApplication> getApplications()
        {
            return this.panuiApplications;
        }
        public List<ShareTransfer> getShareTrans()
        {
            return this.shareTransfers;
        }
    }
}
