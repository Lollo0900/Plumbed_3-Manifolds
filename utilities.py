import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import random


def RandomPlumbing():
    n = np.random.randint(1, 25 + 1)
    weights = np.random.randint(-20, 20 + 1, size=n)
    genera = np.random.randint(-4, 4 + 1, size=n)
    disks_vec = np.random.randint(0, 2 + 1, size=n)
    G = nx.MultiGraph()
    for i in range(n):
        G.add_node(i, weight=weights[i], genus=genera[i], disks=disks_vec[i])
    for i in range(n):
        j = np.random.randint(i, n)
        G.add_edge(i, j, orientation=random.choice([1, -1]))
        if i == j and i != n - 1:
            j = np.random.randint(i + 1, n)
            G.add_edge(i, j, orientation=random.choice([1, -1]))
    return G


def plot_graph(G):
    # Create a layout for the nodes
    pos = nx.spring_layout(G)
    # Assign weights as labels to nodes
    labels = {node: str(data.get('weight')) + '\n' + '[' + str(data.get('genus')) + ',' + str(data.get('disks')) + ']'
              for node, data in G.nodes(data=True)}
    edge_colors = ['blue' if G[u][v][keys]['orientation'] == +1 else 'red' for u, v, keys in G.edges(keys=True)]
    # Draw the graph with vertex labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=900)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color='black')
    # nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue',
    # edge_color=edge_colors, node_size=850, font_size=8)
    # Display the graph
    plt.show()


def GenusCalculus(g1, g2):
    g = None
    if g1 * g2 >= 0:
        g = g1 + g2
    elif g1 > 0 and g2 < 0:
        g = -2 * g1 + g2
    elif g1 < 0 and g2 > 0:
        g = g1 - 2 * g2
    return g


def R0_reversal(G, v):
    done = None
    for u, v, key in G.edges(v, keys=True):
        if u == v and G.nodes[v]['genus'] >= 0:
            pass
        else:
            G[u][v][key]['orientation'] = -G[u][v][key]['orientation']
    done = True
    return G, done


def R1_blowdownA(G, v):
    done = False
    neighbors_list = list(G.neighbors(v))
    neighbors_not_self = [x for x in neighbors_list if x != v]
    if abs(G.nodes[v]['weight']) == 1 and G.nodes[v]['genus'] == 0 and G.nodes[v]['disks'] == 0 and (
            len(G.edges(v)) == 1 and len(neighbors_not_self) == 1):
        for g in G.neighbors(v): G.nodes[g]['weight'] -= G.nodes[v]['weight']
        G.remove_node(v)
        done = True
    return G, done


def R1_blowupA(G, v):
    done = False
    n = len(G.nodes)
    G.add_node(n, weight=random.choice([-1, 1]), genus=0, disks=0)
    G.nodes[v]['weight'] += G.nodes[n]['weight']
    G.add_edge(n, v, key=0, orientation=random.choice([-1, 1]))
    return G, done


def R1_blowdownB(G, v):
    done = False
    neighbors_list = list(G.neighbors(v))
    if abs(G.nodes[v]['weight']) == 1 and G.nodes[v]['genus'] == 0 and G.nodes[v]['disks'] == 0 and len(
            neighbors_list) == 2 and v not in neighbors_list:
        new_edge_orientation = -G.nodes[v]['weight'] * G[neighbors_list[0]][v][0]['orientation'] * G[neighbors_list[1]][v][0]['orientation']
        l = 0
        for u, v, key in G.edges(neighbors_list[0], keys=True):
            if u == neighbors_list[0] and v == neighbors_list[1]:
                l += 1
        G.add_edge(neighbors_list[0], neighbors_list[1], key=l, orientation=new_edge_orientation)
        G.nodes[neighbors_list[0]]['weight'] -= G.nodes[v]['weight']
        G.nodes[neighbors_list[1]]['weight'] -= G.nodes[v]['weight']
        G.remove_node(v)
        done = True
    return G, done


def R1_blowupB(G, v):
    done = None
    neighbors_list = list(G.neighbors(v))
    neighbors_not_self = [x for x in neighbors_list if x != v]
    if len(neighbors_not_self) >= 1:
        r = random.choice(neighbors_not_self)
        l = -1
        for u, n, key in G.edges(v, keys=True):
            if u == v and n == r:
                l += 1
        if l > 0:
            done = False
        else:
            n = len(G.nodes)
            G.add_node(n, weight=random.choice([-1, 1]), genus=0, disks=0)
            G.nodes[v]['weight'] += G.nodes[n]['weight']
            G.nodes[r]['weight'] += G.nodes[n]['weight']
            e0 = G[v][r][0]['orientation']
            G.remove_edge(v, r, key=0)
            e1 = random.choice([-1, 1])
            e2 = -e0 * e1 * G.nodes[n]['weight']
            G.add_edge(v, n, key=0, orientation=e1)
            G.add_edge(r, n, key=0, orientation=e2)
            done = True
    else:
        done = False
    return G, done


def R1_blowdownC(G, v):
    done = None
    neighbors_list = list(G.neighbors(v))
    neighbors_not_self = [x for x in neighbors_list if x != v]
    if len(neighbors_not_self) == 1 and len(neighbors_list) == 1 and abs(G.nodes[v]['weight']) == 1 and abs(
            G.nodes[v]['genus']) + abs(G.nodes[v]['disks']) == 0 and len(G.edges(v)) == 2:
        e0 = -G[v][neighbors_not_self[0]][0]['orientation'] * G[v][neighbors_not_self[0]][1]['orientation'] * G.nodes[v]['weight']
        G.nodes[neighbors_not_self[0]]['weight'] -= 2 * G.nodes[v]['weight']
        G.remove_node(v)
        G.add_edge(neighbors_not_self[0], neighbors_not_self[0], orientation=e0)
        done = True
    else:
        done = False
    return G, done


def R1_blowupC(G, v):
    done = None
    if v in list(G.neighbors(v)):
        n = len(G.nodes)
        e0 = G[v][v][0]['orientation']
        e = random.choice([-1, 1])
        G.nodes[v]['weight'] += 2 * e
        e1 = random.choice([-1, 1])
        e2 = -e * e0 * e1
        G.remove_edge(v, v, key=0)
        G.add_node(n, weight=e, genus=0, disks=0)
        G.add_edge(v, n, orientation=e1)
        G.add_edge(v, n, orientation=e2)
        done = True
    else:
        done = False
    return G, done


def R2_RP2_extrusion(G, v):
    done = None
    if G.nodes[v]['genus'] < 0:
        n = len(G.nodes)
        G.nodes[v]['genus'] += 1
        d1 = random.choice([-1, 1])
        d2 = random.choice([-1, 1])
        delta = (d1 + d2) / 2
        G.add_node(n, weight=delta, genus=0, disks=0)
        G.add_node(n + 1, weight=2 * d1, genus=0, disks=0)
        G.add_node(n + 2, weight=2 * d2, genus=0, disks=0)
        G.add_edge(v, n, orientation=random.choice([-1, 1]))
        G.add_edge(n, n + 1, orientation=random.choice([-1, 1]))
        G.add_edge(n, n + 2, orientation=random.choice([-1, 1]))
        done = True
    else:
        done = False
    return G, done


def R3_absorbtion(G, v):
    done = None
    neighbors_list = list(G.neighbors(v))
    if abs(G.nodes[v]['weight']) + abs(G.nodes[v]['genus']) + abs(G.nodes[v]['disks']) == 0 and len(
            neighbors_list) == 2 and v not in neighbors_list:
        e = G[neighbors_list[0]][v][0]['orientation']
        ebar = G[neighbors_list[1]][v][0]['orientation']
        G.remove_node(v)
        for u, v, key in G.edges(neighbors_list[1], keys=True):
            if u == v or (u == neighbors_list[1] and v == neighbors_list[0]):
                G.add_edge(neighbors_list[0], neighbors_list[0], orientation=G[u][v][0]['orientation'])
            else:
                G.add_edge(neighbors_list[0], v, orientation=-e * ebar * G[u][v][0]['orientation'])
        G.nodes[neighbors_list[0]]['weight'] += G.nodes[neighbors_list[1]]['weight']
        G.nodes[neighbors_list[0]]['genus'] = GenusCalculus(G.nodes[neighbors_list[0]]['genus'],
                                                            G.nodes[neighbors_list[1]]['genus'])
        G.nodes[neighbors_list[0]]['disks'] += G.nodes[neighbors_list[1]]['disks']
        G.remove_node(neighbors_list[1])
        done = True
    else:
        done = False
    return G, done


def R4andR5_absorption(G, v):
    done = None
    neighbors_list = list(G.neighbors(v))
    neighbors_not_self = [x for x in neighbors_list if x != v]
    if len(neighbors_not_self) == 1 and len(neighbors_list) == 1 and abs(G.nodes[v]['weight']) + abs(
            G.nodes[v]['genus']) + abs(G.nodes[v]['disks']) == 0 and len(G.edges(v)) == 2:
        if G[v][neighbors_not_self[0]][0]['orientation'] * G[v][neighbors_not_self[0]][1]['orientation'] == 1:
            G.nodes[neighbors_not_self[0]]['genus'] = GenusCalculus(G.nodes[neighbors_not_self[0]]['genus'], -2)
        else:
            G.nodes[neighbors_not_self[0]]['genus'] = GenusCalculus(G.nodes[neighbors_not_self[0]]['genus'], +1)
        G.remove_node(v)
        done = True
    else:
        done = False
    return G, done


def R4_extrusion(G, v):
    done = None
    if G.nodes[v]['genus'] <= -2:
        n = len(G.nodes)
        G.add_node(n, weight=0, genus=0, disks=0)
        e = random.choice([-1, 1])
        G.add_edge(v, n, orientation=e)
        G.add_edge(v, n, orientation=e)
        G.nodes[v]['genus'] += 2
        done = True
    else:
        done = False
    return G, done


def R5_extrusion(G, v):
    done = None
    if G.nodes[v]['genus'] <= -3 or G.nodes[v]['genus'] >= 1:
        n = len(G.nodes)
        G.add_node(n, weight=0, genus=0, disks=0)
        e = random.choice([-1, 1])
        G.add_edge(v, n, orientation=e)
        G.add_edge(v, n, orientation=-e)
        if G.nodes[v]['genus'] >= 1:
            G.nodes[v]['genus'] -= 1
        elif G.nodes[v]['genus'] <= -3:
            G.nodes[v]['genus'] += 2
        done = True
    else:
        done = False
    return G, done


def R8_annulus_absorbtion(G, v):
    done = False
    neighbors_list = list(G.neighbors(v))
    neighbors_not_self = [x for x in neighbors_list if x != v]
    if abs(G.nodes[v]['genus']) == 0 and G.nodes[v]['disks'] == 1 and (
            len(G.edges(v)) == 1 and len(neighbors_not_self) == 1):
        G.nodes[neighbors_not_self[0]]['disks'] += 1
        G.remove_node(v)
        done = True
    return G, done


def R8_annulus_extrusion(G, v):
    done = False
    if G.nodes[v]['disks'] >= 1:
        n = len(G.nodes)
        G.add_node(n, weight=0, genus=0, disks=1)
        G.nodes[v]['disks'] -= 1
        G.add_edge(n, v, key=0, orientation=random.choice([-1, 1]))
        done = True
    return G, done


def RandomNeumannMove(G):
    v = random.choice(list(G.nodes))
    move = random.choice(["R0", "R1a", "R1b", "R1c", "R2", "R3", "R4", "R5", "R8"])
    updown = random.choice([True, False])

    result = None

    if move == "R0":
        G, result = R0_reversal(G, v)
    elif move == "R1a":
        if updown:
            G, result = R1_blowupA(G, v)
        else:
            G, result = R1_blowdownA(G, v)
    elif move == "R1b":
        if updown:
            G, result = R1_blowupB(G, v)
        else:
            G, result = R1_blowdownB(G, v)
    elif move == "R1a":
        if updown:
            G, result = R1_blowupC(G, v)
        else:
            G, result = R1_blowdownC(G, v)
    elif move == "R2":
        G, result = R2_RP2_extrusion(G, v)
    elif move == "R3":
        G, result = R3_absorbtion(G, v)
    elif move == "R4":
        if updown:
            G, result = R4_extrusion(G, v)
        else:
            G, result = R4andR5_absorption(G, v)
    elif move == "R5":
        if updown:
            G, result = R5_extrusion(G, v)
        else:
            G, result = R4andR5_absorption(G, v)
    elif move == "R8":
        if updown:
            G, result = R8_annulus_extrusion(G, v)
        else:
            G, result = R8_annulus_absorbtion(G, v)

    return G, result


def EquivPair(Nmax):
    n1 = np.random.randint(1, Nmax + 1)
    G = RandomPlumbing()
    G1 = G.copy()
    result = None
    result1 = None
    for i in range(n1):
        G, result = RandomNeumannMove(G)
    n2 = np.random.randint(1, Nmax + 1)
    for i in range(n2):
        G1, result1 = RandomNeumannMove(G1)
    return G, G1, True


def InequivPair(Nmax):
    n1 = np.random.randint(1, Nmax + 1)
    result = None
    result1 = None
    G = RandomPlumbing()
    G1 = RandomPlumbing()
    for i in range(n1):
        G, result = RandomNeumannMove(G)
    n2 = np.random.randint(1, Nmax + 1)
    for i in range(n2):
        G1, result1 = RandomNeumannMove(G1)
    return G, G1, False


def TweakPair(Nmax):
    result = None
    result1 = None
    n1 = np.random.randint(1, Nmax + 1)
    G = RandomPlumbing()
    G1 = G.copy()
    v = random.choice(list(G1.nodes))
    t = np.random.randint(1, 3 + 1) * random.choice([1, -1])
    G1.nodes[v]['weight'] += t
    for i in range(n1):
        G, result = RandomNeumannMove(G)
    n2 = np.random.randint(1, Nmax + 1)
    for i in range(n2):
        G1, result1 = RandomNeumannMove(G1)
    return G, G1, False
