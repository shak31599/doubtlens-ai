"""Generate transparently labeled, original synthetic sample questions; not real user data."""
from pathlib import Path
import csv
base=Path(__file__).resolve().parents[1]
topics={
'DSA':[
('binary search','sorted arrays and boundary indices'),('Dijkstra algorithm','negative edge weights'),('BFS traversal','shortest paths in an unweighted graph'),('DFS traversal','cycle detection'),('merge sort','time and space complexity'),('quick sort','worst case pivot choice'),('hash tables','collision handling'),('binary trees','height balancing'),('dynamic programming','overlapping subproblems'),('recursion','base cases and stack overflow'),('heap','priority queues'),('linked lists','pointer updates'),('union find','path compression'),('topological sort','dependency graphs'),('sliding window','longest substring'),('two pointers','sorted array pairs'),('segment tree','range sum queries'),('trie','prefix lookup'),('minimum spanning tree','Kruskal versus Prim'),('graph adjacency list','memory complexity'),('backtracking','pruning search trees'),('memoization','cache lookup'),('Kadane algorithm','maximum subarray sum'),('prefix sum','fast interval queries'),('stack','balanced parentheses'),('queue','breadth first traversal'),('red black tree','rotations'),('Floyd Warshall','all pairs shortest paths'),('bit manipulation','XOR tricks'),('greedy algorithms','local optimal choices')],
'Web Development':[
('CSS flexbox','horizontal alignment'),('CSS grid','responsive layout'),('JavaScript promises','asynchronous chaining'),('React hooks','state updates'),('REST APIs','HTTP status codes'),('HTML forms','input validation'),('CORS','browser preflight requests'),('cookies','session storage'),('JWT authentication','token expiry'),('Node.js','event loop'),('Express middleware','request pipeline'),('WebSockets','real time chat'),('TypeScript','type inference'),('CSS media queries','mobile breakpoints'),('DOM events','event bubbling'),('fetch API','network error handling'),('Next.js','server side rendering'),('React components','rerender behavior'),('accessibility ARIA','screen reader labels'),('browser caching','cache headers'),('GraphQL','query resolvers'),('HTTP headers','content security policy'),('OAuth','authorization redirects'),('database migrations','web app deployments'),('frontend bundling','tree shaking'),('service workers','offline caching'),('CSS specificity','overridden styles'),('HTML semantics','accessible headings'),('web performance','lazy loading'),('API pagination','cursor-based endpoints')],
'Machine Learning':[
('gradient descent','learning rate schedules'),('overfitting','regularization strength'),('cross validation','train test leakage'),('random forests','feature importance'),('logistic regression','probability calibration'),('neural networks','backpropagation'),('transformer attention','query key value vectors'),('word embeddings','semantic similarity'),('class imbalance','precision and recall'),('confusion matrices','false positive rates'),('PCA','explained variance'),('k means clustering','cluster assignment'),('decision trees','information gain'),('batch normalization','training stability'),('dropout','generalization'),('learning curves','bias variance tradeoff'),('tokenization','subword handling'),('sentence transformers','vector retrieval'),('fine tuning','frozen layers'),('XGBoost','boosted learners'),('SVM','kernel margins'),('feature scaling','standardization'),('data augmentation','training robustness'),('ROC AUC','threshold choice'),('Naive Bayes','conditional independence'),('embedding dimensions','cosine distance'),('early stopping','validation loss'),('hyperparameter search','random search'),('model drift','distribution changes'),('imbalanced classification','macro F1')],
'Core Computer Science':[
('process scheduling','round robin'),('deadlocks','resource allocation'),('virtual memory','page faults'),('operating systems','context switching'),('SQL indexes','query plans'),('database normalization','functional dependencies'),('transactions','ACID guarantees'),('TCP handshake','connection setup'),('UDP protocols','packet delivery'),('DNS resolution','hostname lookup'),('OSI model','network layers'),('thread synchronization','race conditions'),('mutex locks','critical sections'),('cache coherence','multicore processors'),('filesystem inodes','file metadata'),('memory management','heap allocation'),('compiler parsing','abstract syntax trees'),('garbage collection','reference counting'),('network routing','IP subnets'),('database isolation','phantom reads'),('CPU pipelines','instruction hazards'),('distributed systems','consensus'),('CAP theorem','partition tolerance'),('public key cryptography','digital signatures'),('TLS','certificate verification'),('binary representation','two complement'),('system calls','kernel mode transitions'),('semaphores','producer consumer'),('relational algebra','join operations'),('paging algorithms','LRU eviction')]
}
templates=[
'Why does {topic} behave differently when dealing with {detail}?',
'How should I debug a problem involving {topic} and {detail}?',
'Can you explain the tradeoffs of {topic} for {detail} with an example?',
'What common mistakes should I avoid when using {topic} for {detail}?',
'How can I test whether my implementation of {topic} handles {detail} correctly?'
]
rows=[]
for category,concepts in topics.items():
    for j,(topic,detail) in enumerate(concepts):
        for k,template in enumerate(templates):
            rows.append({'question_id':f'syn-{len(rows)+1:04d}','question_text':template.format(topic=topic,detail=detail),'category':category,'source':'Original synthetic examples','source_url':'','license':'CC0-1.0 (project-authored synthetic examples)'})
out=base/'data/samples/synthetic_questions.csv'
with out.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
print('Synthetic rows:',len(rows))
