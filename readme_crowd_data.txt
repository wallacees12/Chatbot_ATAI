General Description 


The file “ATAI_crowd_data.tsv” provides a simulated result of a set crowdsourcing microtasks (HITs) implemented as different batches. These microtasks correspond to the Verification and Validation (VV) type of tasks presented in the lecture: the crowd is asked to indicate if a piece of information ---an RDF triple (subject,predicate,object)--- is correct or not. 




Attributes of the TSV File


* HITId: identifier of the microtask 
* HITTypeId: identifier of the batch of microtasks (or HIT template)        
* Title: title of the microtask         
* Reward: economic reward to be paid for one assignment (answer) to the microtask
* AssignmentId: identifier of the assignment (answer) provided by the worker for this microtask
* WorkerId: identifier of the worker in the platform        
* AssignmentStatus: status of the assignment        
* WorkTimeInSeconds: time measured in seconds that the worker took to complete the microtask        
* LifetimeApprovalRate: reputation of the worker measured by the approval rate accumulated over time in the platform
* Input1ID: subject of the RDF Triple shown assessed in this microtask        
* Input2ID: predicate of the RDF Triple shown assessed in this microtask        
* Input3ID: object of the RDF Triple shown assessed in this microtask        
* AnswerID: identifier of the type of answer provided by the worker        
* AnswerLabel: text of the answer provided by the worker. The possible values for this attribute are “CORRECT” and “INCORRECT”.        
* FixPosition: when the answer provided by the worker is “INCORRECT”, the worker has the opportunity to include details on the way the triple should be fixed. FixPosition indicates the position of the RDF triple (subject, predicate, object) that the worker thinks needs to be corrected.
* FixValue: FixValue indicates the value that would make the RDF triple correct, if the value in the position indicated as FixPosition were exchanged with this one. If the worker recognizes that a concrete position of the RDF triple is wrong, but does not know the correct value for it, the worker may leave FixValue empty after providing a value for FixPosition.




Implementing the Crowdsourcing Capability in Your Agent


Your agent is expected to process the crowdsourced data and merge it with the initial knowledge graph. Given that we ask the crowd to augment and correct the data in the knowledge graph, the crowdsourced data provided here may extend and update the initial knowledge graph. In order to do so, your agent will need to:


1. Filter out individual answers from malicious crowd workers. Crowd workers can be malicious for a variety of reasons, as explained in the lecture.
2. Aggregate individual crowd answers with majority voting, in order to obtain one single final answer per microtask (or HIT).
3. Compute the inter-rater agreement (Fleiss’ kappa) per batch. This agreement value should be indicated in the conversation when the answer provided by the agent was obtained from the crowd. Additionally, when providing such answers that were obtained from the crowd, the agent is expected to indicate the concrete distribution of answers for that specific microtask. 




Example: 


Human Assessor: What is the birthplace of Christopher Nolan? 


Agent: London - according to the crowd, who had an inter-rater agreement of 0.72 in this batch. 
The answer distribution for this specific task was 2 support votes and 1 reject vote. 


In the report, please also provide user engagement statistics.