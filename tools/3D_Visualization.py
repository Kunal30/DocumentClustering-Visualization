import plotly as py
import plotly.graph_objs as go
import pickle
import numpy as np

# x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 400).transpose()
color=['blue','green','red','cyan',
'magenta','yellow','black','#293f63',
'#2a8c25','#c6c431','#aa317e','#68271c',
'#f27900','#0be09c','#ba9e9e','#a31f01',
'#42b765','#1c98a8','#32e0ff','#a5568f']


file = open('latent.txt', 'r')
latent = pickle.load(file)

f = open('Z.txt', 'r')
Z = pickle.load(f)

f = open('topics.txt', 'r')
topics = pickle.load(f)

f = open('topics.pickle', 'r')
topics_dist = pickle.load(f)

# print(topics_dist)

tsne_x=latent[:,0]
tsne_y=latent[:,1]

topics=np.reshape(topics,(topics.shape[0],))

tsne_z=np.zeros((topics.shape[0],))
color_docx=[]

for i in range(topics.shape[0]):
    tsne_z[i]=Z[int(topics[i])]
    color_docx.append(color[int(topics[i])] )

word_labels=[]
for i in range(topics.shape[0]):
	word_labels.append(topics_dist[int(topics[i])])
# Just need to replace x, y, z with tsne_x, tsne_y, tsne_z


# # print('x=',x.shape)
# # print('y=',y.shape)
# print('z=',z.shape)
trace1 = go.Scatter3d(
    x=tsne_x,
    y=tsne_y,
    z=tsne_z,
    text=word_labels,
    mode='markers',
    marker=dict(
        size=12,
        color=color_docx,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename='3d-scatter-colorscale')
print('Open your Browser')