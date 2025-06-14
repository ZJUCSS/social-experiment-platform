"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: retrieve.py
Description: This defines the "Retrieve" module for generative agents.
"""

import sys

from utils import *
from persona.prompt_template.gpt_structure import *

from numpy import dot
from numpy.linalg import norm


def retrieve(persona, perceived):
    """
    This function takes the events that are perceived by the persona as input
    and returns a set of related events and thoughts that the persona would
    need to consider as context when planning.

    INPUT:
      perceived: a list of event <ConceptNode>s that represent any of the events
      `         that are happening around the persona. What is included in here
                are controlled by the att_bandwidth and retention
                hyper-parameters.
    OUTPUT:
      retrieved: a dictionary of dictionary. The first layer specifies an event,
                 while the latter layer specifies the "curr_event", "events",
                 and "thoughts" that are relevant.
    """
    # We rerieve events and thoughts separately.
    retrieved = dict()
    for event in perceived:
        retrieved[event.description] = dict()
        retrieved[event.description]["curr_event"] = event

        relevant_events = persona.a_mem.retrieve_relevant_events(event.subject, event.predicate, event.object)
        retrieved[event.description]["events"] = list(relevant_events)

        relevant_thoughts = persona.a_mem.retrieve_relevant_thoughts(event.subject, event.predicate, event.object)
        retrieved[event.description]["thoughts"] = list(relevant_thoughts)

    return retrieved


def cos_sim(a, b):
    """
    This function calculates the cosine similarity between two input vectors
    'a' and 'b'. Cosine similarity is a measure of similarity between two
    non-zero vectors of an inner product space that measures the cosine
    of the angle between them.

    INPUT:
      a: 1-D array object
      b: 1-D array object
    OUTPUT:
      A scalar value representing the cosine similarity between the input
      vectors 'a' and 'b'.

    Example input:
      a = [0.3, 0.2, 0.5]
      b = [0.2, 0.2, 0.5]
    """
    return dot(a, b) / (norm(a) * norm(b))


def normalize_dict_floats(d, target_min, target_max):
    """
    This function normalizes the float values of a given dictionary 'd' between
    a target minimum and maximum value. The normalization is done by scaling the
    values to the target range while maintaining the same relative proportions
    between the original values.

    INPUT:
      d: Dictionary. The input dictionary whose float values need to be
         normalized.
      target_min: Integer or float. The minimum value to which the original
                  values should be scaled.
      target_max: Integer or float. The maximum value to which the original
                  values should be scaled.
    OUTPUT:
      d: A new dictionary with the same keys as the input but with the float
         values normalized between the target_min and target_max.

    Example input:
      d = {'a':1.2,'b':3.4,'c':5.6,'d':7.8}
      target_min = -5
      target_max = 5
    """
    min_val = min(val for val in d.values())  # 没东西？
    max_val = max(val for val in d.values())
    range_val = max_val - min_val

    if range_val == 0:
        for key, val in d.items():
            d[key] = (target_max - target_min) / 2
    else:
        for key, val in d.items():
            d[key] = (val - min_val) * (target_max - target_min) / range_val + target_min
    return d


def top_highest_x_values(d, x):
    """
    This function takes a dictionary 'd' and an integer 'x' as input, and
    returns a new dictionary containing the top 'x' key-value pairs from the
    input dictionary 'd' with the highest values.

    INPUT:
      d: Dictionary. The input dictionary from which the top 'x' key-value pairs
         with the highest values are to be extracted.
      x: Integer. The number of top key-value pairs with the highest values to
         be extracted from the input dictionary.
    OUTPUT:
      A new dictionary containing the top 'x' key-value pairs from the input
      dictionary 'd' with the highest values.

    Example input:
      d = {'a':1.2,'b':3.4,'c':5.6,'d':7.8}
      x = 3
    """
    top_v = dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:x])
    return top_v


def extract_recency(persona, nodes):
    """
    Gets the current Persona object and a list of nodes that are in a
    chronological order, and outputs a dictionary that has the recency score
    calculated.

    INPUT:
      persona: Current persona whose memory we are retrieving.
      nodes: A list of Node object in a chronological order.
    OUTPUT:
      recency_out: A dictionary whose keys are the node.node_id and whose values
                   are the float that represents the recency score.
    """
    recency_vals = [persona.scratch.recency_decay**i for i in range(1, len(nodes) + 1)]

    recency_out = dict()
    for count, node in enumerate(nodes):
        recency_out[node.node_id] = recency_vals[count]

    return recency_out


def extract_importance(persona, nodes):
    """
    Gets the current Persona object and a list of nodes that are in a
    chronological order, and outputs a dictionary that has the importance score
    calculated.

    INPUT:
      persona: Current persona whose memory we are retrieving.
      nodes: A list of Node object in a chronological order.
    OUTPUT:
      importance_out: A dictionary whose keys are the node.node_id and whose
                      values are the float that represents the importance score.
    """
    importance_out = dict()
    for count, node in enumerate(nodes):
        importance_out[node.node_id] = node.poignancy

    return importance_out


def extract_relevance(persona, nodes, focal_pt):
    """
    Gets the current Persona object, a list of nodes that are in a
    chronological order, and the focal_pt string and outputs a dictionary
    that has the relevance score calculated.

    INPUT:
      persona: Current persona whose memory we are retrieving.
      nodes: A list of Node object in a chronological order.
      focal_pt: A string describing the current thought of revent of focus.
    OUTPUT:
      relevance_out: A dictionary whose keys are the node.node_id and whose values
                   are the float that represents the relevance score.
    """
    focal_embedding = get_embedding(focal_pt)

    relevance_out = dict()
    for count, node in enumerate(nodes):
        node_embedding = persona.a_mem.embeddings[node.node_id]
        relevance_out[node.node_id] = cos_sim(node_embedding, focal_embedding)

    return relevance_out


def new_retrieve(persona, focal_points, n_count=30):
    """
    Given the current persona and focal points (focal points are events or
    thoughts for which we are retrieving), we retrieve a set of nodes for each
    of the focal points and return a dictionary.

    INPUT:
      persona: The current persona object whose memory we are retrieving.
      focal_points: A list of focal points (string description of the events or
                    thoughts that is the focus of current retrieval).
    OUTPUT:
      retrieved: A dictionary whose keys are a string focal point, and whose
                 values are a list of Node object in the agent's associative
                 memory.

    Example input:
      persona = <persona> object
      focal_points = ["How are you?", "Jane is swimming in the pond"]
    """
    # <retrieved> is the main dictionary that we are returning
    retrieved = dict()
    for focal_pt in focal_points:
        # Getting all nodes from the agent's memory (both thoughts and events) and
        # sorting them by the datetime of creation.
        # You could also imagine getting the raw conversation, but for now.
        L.debug(f"seq events and seq_thought: {persona.a_mem.get_events() + persona.a_mem.get_thoughts()}")
        nodes = [
            [i.last_accessed, i]
            for i in persona.a_mem.get_events() + persona.a_mem.get_thoughts()
            if "idle" not in i.embedding_key
        ]
        L.debug(f"{nodes}")
        nodes = sorted(nodes, key=lambda x: x[0])
        nodes = [i for created, i in nodes]

        # Calculating the component dictionaries and normalizing them.
        print("persona: ", persona)
        print("nodes: ", nodes)
        recency_out = extract_recency(persona, nodes)
        L.debug(f"recency_out: {recency_out}")
        recency_out = normalize_dict_floats(recency_out, 0, 1)
        importance_out = extract_importance(persona, nodes)
        importance_out = normalize_dict_floats(importance_out, 0, 1)
        relevance_out = extract_relevance(persona, nodes, focal_pt)
        # lg: cos_sim()
        relevance_out = normalize_dict_floats(relevance_out, 0, 1)

        # Computing the final scores that combines the component values.
        # Note to self: test out different weights. [1, 1, 1] tends to work
        # decently, but in the future, these weights should likely be learned,
        # perhaps through an RL-like process.
        # gw = [1, 1, 1]
        # gw = [1, 2, 1]
        gw = [0.5, 3, 2]
        master_out = dict()
        for key in recency_out.keys():
            master_out[key] = (
                persona.scratch.recency_w * recency_out[key] * gw[0]
                + persona.scratch.relevance_w * relevance_out[key] * gw[1]
                + persona.scratch.importance_w * importance_out[key] * gw[2]
            )

        master_out = top_highest_x_values(master_out, len(master_out.keys()))
        # for key, val in master_out.items():
        #     print(persona.a_mem.id_to_node[key].embedding_key, val)
        #     print(
        #         persona.scratch.recency_w * recency_out[key] * 1,
        #         persona.scratch.relevance_w * relevance_out[key] * 1,
        #         persona.scratch.importance_w * importance_out[key] * 1,
        #     )

        # Extracting the highest x values.
        # <master_out> has the key of node.id and value of float. Once we get the
        # highest x values, we want to translate the node.id into nodes and return
        # the list of nodes.
        master_out = top_highest_x_values(master_out, n_count)
        master_nodes = [persona.a_mem.nodes[key] for key in list(master_out.keys())]

        for n in master_nodes:
            n.last_accessed = persona.scratch.curr_time

        retrieved[focal_pt] = master_nodes

    return retrieved


#
# We should improve this function further and further - .
def retrieve_dai(persona, perceived):
    """
    This function takes the events that are perceived by the persona as input
    and returns a set of related events and thoughts that the persona would
    need to consider as context when planning.
    此函数将角色感知到的事件作为输入，并返回一组相关事件和想法，角色在计划时需要将其作为上下文考虑。
    INPUT:
       perceived: a list of event <ConceptNode>s that represent any of the events
       `         that are happening around the persona. What is included in here
                 are controlled by the att_bandwidth and retention
                 hyper-parameters.
    OUTPUT:
       retrieved: a dictionary of dictionary. The first layer specifies an event,
                  while the latter layer specifies the "curr_event", "events",
                  and "thoughts" that are relevant.
    """
    # We rerieve events and thoughts separately.
    retrieved = dict()
    for event_name, perceived in perceived.items():
        for event in perceived:
            retrieved[event_name] = dict()
            retrieved[event_name]["curr_event"] = event

            relevant_events = persona.a_mem.retrieve_relevant_events(event.subject, event.predicate, event.object)
            # print("pig-------------------------------")
            # print(len(relevant_events))
            # retrieved[event.description]["events"] = list(relevant_events)

            relevant_thoughts = persona.a_mem.retrieve_relevant_thoughts(event.subject, event.predicate, event.object)
            retrieved[event_name]["events"] = list(relevant_events) + list(relevant_thoughts)
            retrieved[event_name]["thoughts"] = list()

    return retrieved


def retrieve_dai_custom(persona, perceived):
    """
    This function takes the events that are perceived by the persona as input
    and returns a set of related events and thoughts that the persona would
    need to consider as context when planning.
    此函数将角色感知到的事件作为输入，并返回一组相关事件和想法，角色在计划时需要将其作为上下文考虑。
    INPUT:
       perceived: a list of event <ConceptNode>s that represent any of the events
       `         that are happening around the persona. What is included in here
                 are controlled by the att_bandwidth and retention
                 hyper-parameters.
    OUTPUT:
       retrieved: a dictionary of dictionary. The first layer specifies an event,
                  while the latter layer specifies the "curr_event", "events",
                  and "thoughts" that are relevant.
    """
    # We rerieve events and thoughts separately.
    retrieved = dict()
    for event_name, perceived in perceived.items():
        for event in perceived:
            retrieved[event_name] = dict()
            retrieved[event_name]["curr_event"] = event

            task_info = persona.get_workflow_stage_config()["plan"]
            task_description = task_info.get("task", "Decide the next action.")

            desc_embedding = get_embedding(event.description)
            task_embedding = get_embedding(task_description)
            relevant_events = persona.a_mem.retrieve_by_sim("event", desc_embedding, 10)

            relevant_thoughts = persona.a_mem.retrieve_by_sim("thought", desc_embedding, 10)

            retrieved[event_name]["events"] = list(relevant_events) + list(relevant_thoughts)
            retrieved[event_name]["thoughts"] = list()

            relevant_speeches = persona.a_mem.retrieve_by_sim("archive", task_embedding, 10)

            retrieved[event_name]["speeches"] = list(relevant_speeches)

    return retrieved


#
# We should improve this function further and further - .
def new_retrieve_dai(persona, retrieved, n_count=30):
    """
    Given the current persona and focal points (focal points are events or
    thoughts for which we are retrieving), we retrieve a set of nodes for each
    of the focal points and return a dictionary.

    INPUT:
      persona: The current persona object whose memory we are retrieving.
      focal_points: A list of focal points (string description of the events or
                    thoughts that is the focus of current retrieval).
    OUTPUT:
      retrieved: A dictionary whose keys are a string focal point, and whose
                 values are a list of Node object in the agent's associative
                 memory.

    Example input:
      persona = <persona> object
      focal_points = ["How are you?", "Jane is swimming in the pond"]
    """
    # <retrieved> is the main dictionary that we are returning
    # Getting all nodes from the agent's memory (both thoughts and events) and
    # sorting them by the datetime of creation.
    # You could also imagine getting the raw conversation, but for now.
    """
persona.a_mem.seq_event + persona.a_mem.seq_thought：通过将事件节点列表和思想节点列表连接起来，我们得到了一个包含了所有事件和思想节点的混合列表。
for i in ... if "idle" not in i.embedding_key：在遍历这个混合列表时，排除了那些嵌入键中包含字符串"idle"的节点，因为它们是表示空闲状态的特殊节点。
[[i.last_accessed, i] for i in ...]：在遍历列表时，我们将每个节点 i 与其最后访问时间 i.last_accessed 组成一个二元组，并将这些二元组组成一个新的列表。这样做是为了方便后续的按照最后访问时间排序。
    
    """
    # nodes = [[i.last_accessed, i]
    #             for i in old_retrieved["events"]
    #             if "idle" not in i.embedding_key]
    new_retrieved = retrieved.copy()
    for keys, vals in new_retrieved.items():
        if vals["events"] == []:
            print("--ok")
            continue
        nodes = []
        for event in vals["events"]:
            if "idle" not in event.embedding_key:
                nodes.append([event.last_accessed, event])

        nodes = sorted(nodes, key=lambda x: x[0])
        nodes = [i for created, i in nodes]

        # Calculating the component dictionaries and normalizing them.
        recency_out = extract_recency(persona, nodes)
        recency_out = normalize_dict_floats(recency_out, 0, 1)
        importance_out = extract_importance(persona, nodes)
        importance_out = normalize_dict_floats(importance_out, 0, 1)

        # Computing the final scores that combines the component values.
        # Note to self: test out different weights. [1, 1, 1] tends to work
        # decently, but in the future, these weights should likely be learned,
        # perhaps through an RL-like process.
        # gw = [1, 1, 1]
        # gw = [1, 2, 1]
        gw = [0.5, 3, 2]
        master_out = dict()
        for key in recency_out.keys():
            master_out[key] = (
                persona.scratch.recency_w * recency_out[key] * gw[0]
                + persona.scratch.importance_w * importance_out[key] * gw[2]
            )
        print("new_retrieve_new_=========")

        master_out = top_highest_x_values(master_out, len(master_out.keys()))
        for key, val in master_out.items():
            print(persona.a_mem.nodes[key].embedding_key, val)
            print(
                persona.scratch.recency_w * recency_out[key] * 1,
                persona.scratch.importance_w * importance_out[key] * 1,
            )

        # Extracting the highest x values.
        # <master_out> has the key of node.id and value of float. Once we get the
        # highest x values, we want to translate the node.id into nodes and return
        # the list of nodes.
        master_out = top_highest_x_values(master_out, n_count)
        master_nodes = [persona.a_mem.nodes[key] for key in list(master_out.keys())]

        for n in master_nodes:
            n.last_accessed = persona.scratch.curr_time

        retrieved[keys]["events"] = list(master_nodes)
        # TODO there is no retrieved thoughts here.

    return retrieved
