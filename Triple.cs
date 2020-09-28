using System;
using System.Collections.Generic;
using System.Text;

namespace Bayesian_Inference
{
    class Triple
    {
        protected List<Relationship> relationshipList;
        protected List<Person> peopleList;
        protected bool isSolved;
        
        public Triple(List<Relationship> relationshipList, List<Person> personList, bool isSolved = false)
        {
            this.relationshipList = relationshipList;
            this.peopleList = personList;
            this.isSolved = isSolved;
        }

        public List<Relationship> getRelationships()
        {
            return this.relationshipList;
        }
        public List<Person> getPeople()
        {
            return this.peopleList;
        }

        public void solve()
        {
            this.isSolved = true;
        }

        public int nParents()
        {
            int n = 0;
            for (int i = 0; i < relationshipList.Count; i++)
            {
                if (relationshipList[i].getIsDeclared() == true)
                {
                    n++;
                }
            }
            return n;
        }

        public List<Relationship> getParents()
        {
            List<Relationship> parents = new List<Relationship>();
            for (int i = 0; i < relationshipList.Count; i++)
            {
                if (relationshipList[i].getIsDeclared() == true)
                {
                    parents.Add(relationshipList[i]);
                }
            }
            return parents;
        }

        public List<Relationship> getChildren()
        {
            List<Relationship> children = new List<Relationship>();
            for (int i = 0; i < relationshipList.Count; i++)
            {
                if (relationshipList[i].getIsDeclared() == false)
                {
                    children.Add(relationshipList[i]);
                }
            }
            return children;
        }

        public void declareChildren()
        {
            for (int i = 0; i < relationshipList.Count; i++)
            {
                if (relationshipList[i].getIsDeclared() == false)
                {
                    relationshipList[i].declare();
                }
            }
        }
    }
}
