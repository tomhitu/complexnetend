from landing_page_lib import *

def main():
    num_nodes1, num_edges1, density1, cc1, degree_values1, degree_histogram1, node_feature_list1, edge_feature_list1, num_nodes2, num_edges2, density2, cc2, degree_values2, degree_histogram2, node_feature_list2, edge_feature_list2 = get_landing_page_result()

    print(f'dataset 1: Chinese Railway Dataset')

    print(f"Number of nodes: {num_nodes1}")
    print(f"Number of edges: {num_edges1}")
    print(f"Number of Density: {density1}")
    print(f"Number of cc: {cc1}")

    print(f"Node feature list: {node_feature_list1}")
    print(f"Edge feature list: {edge_feature_list1}")

    # Plot degree distribution (linear)
    print(f"Network1 degree distribution plot (linear)")
    plt.plot(degree_values1, degree_histogram1, 'o-', linewidth=2)
    print(f"degree_values1: {degree_values1}")
    print(f"degree_histogram1: {degree_histogram1}")
    plt.xlabel('Degree')
    plt.ylabel('Fraction of Nodes')
    plt.title('Degree Distribution')
    plt.show()
    plt.clf()

    # Plot degree distribution (log-log)
    print(f"Network1 degree distribution plot (log-log)")
    plt.loglog(degree_values1, degree_histogram1, 'o-', linewidth=2)
    print(np.power(10, degree_values1))
    print(np.power(10, degree_histogram1))
    plt.xlabel('Degree')
    plt.ylabel('Fraction of Nodes')
    plt.title('Degree Distribution')
    plt.show()
    plt.clf()

    print('--------------------------------------------------------')

    print(f'dataset 2: Paris Multilayer Transport Dataset')
    print(f"Number of nodes: {num_nodes2}")
    print(f"Number of edges: {num_edges2}")
    print(f"Number of Density: {density2}")
    print(f"Number of cc: {cc2}")

    print(f"Node feature list: {node_feature_list2}")
    print(f"Edge feature list: {edge_feature_list2}")

    # Plot degree distribution (linear)
    print(f"Network2 degree distribution plot (linear)")
    plt.plot(degree_values2, degree_histogram2, 'o-', linewidth=2)
    print(f"degree_values2: {degree_values2}")
    print(f"degree_histogram2: {degree_histogram2}")
    plt.xlabel('Degree')
    plt.ylabel('Fraction of Nodes')
    plt.title('Degree Distribution')
    plt.show()
    plt.clf()

    # Plot degree distribution (log-log)
    print(f"Network2 degree distribution plot (log-log)")
    plt.loglog(degree_values2, degree_histogram2, 'o-', linewidth=2)
    print(np.power(10, degree_values2))
    print(np.power(10, degree_histogram2))
    plt.xlabel('Degree')
    plt.ylabel('Fraction of Nodes')
    plt.title('Degree Distribution')
    plt.show()
    plt.clf()

if __name__ == "__main__":
    main()