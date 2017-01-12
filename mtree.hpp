#pragma once

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

template <class T, class S, size_t N> class INode {
public:
    virtual ~INode() {}

    void range(T &query_object, int query_radius, vector<T> &results) {
        range(query_object, query_radius, 0, results);
    }

    void insert(T &insert_object) {
        insert(insert_object, 0);
    }

    virtual void range(T &query_object, int query_radius, int parent_query_distance, vector<T> &results) = 0;
    virtual void insert(T& insert_object, int parent_distance) = 0;
    virtual void set_parent(INode<T, S, N> *parent, int parent_index) = 0;
    virtual void set_entry(int entry, T& object, int parent_distance, INode<T, S, N>* child, int radius) = 0;
    virtual void add_entry(T& object, int parent_distance, INode<T, S, N>* child, int radius) = 0;
};    

template <class T> class IMetric {
public:
    virtual int distance(T &x1, T &x2) = 0;
};

template <class T, class S, size_t N> class InternalNode : public INode<T, S, N> {
public:
    InternalNode(INode<T, S, N> **root, IMetric<T> *metric) {
        this->root = root;
        this->metric = metric;
        
        parent_node = NULL;
        parent_index = -1;
    }
    
    void range(T &query_object, int query_radius, int parent_query_distance, vector<T> &results) {
        for (int i = 0; i < N; i++) {
            T &object = objects[i];
            int radius = radii[i];
            int parent_distance = parent_distances[i];
            
            if (valid[i] && abs(parent_query_distance - parent_distance) <= query_radius + radius) {
                int query_distance = metric->distance(object, query_object);
                if (query_distance <= query_radius + radius) {
                    children[i]->range(query_object, query_radius, query_distance, results);
                }
            }
        }
    }
    
    void insert(T &insert_object, int parent_distance) {
        // Compute distance from each child center to the new object.
        array<int, N> child_distances;
        for (int i = 0; i < N; i++) {
            T &object = objects[i];
            child_distances[i] = valid[i] ? metric->distance(object, insert_object) : INT_MAX;
        }

        // Select the closest child which includes the new object in its radius.
        int min_distance = INT_MAX, child_index = -1;
        for (int i = 0; i < N; i++) {
            if (child_distances[i] < min_distance && child_distances[i] < radii[i]) {
                child_index = i;
                min_distance = child_distances[i];
            }
        }

        if (child_index == -1) {
            // Compute radius size increase for each child.
            array<int, N> radius_increase;
            for (int i = 0; i < N; i++) {
                radius_increase[i] = child_distances[i] - radii[i];
            }

            // Select the child with the minimum radius increase.
            int min_increase = INT_MAX, child_index = -1;
            for (int i = 0; i < N; i++) {
                if (radius_increase[i] < min_increase) {
                    child_index = i;
                    min_increase = radius_increase[i];
                }
            }

            // Increase the child radius.
            radii[child_index] = child_distances[child_index];
        }

        children[child_index]->insert(insert_object, child_distances[child_index]);
    }

    void split() {
        InternalNode<T, S, N> *new_node = new InternalNode<T, S, N>(root, metric);
        array<T, N> objects = this->objects;
        array<INode<T, S, N>*, N> children = this->children;
        array<int, N> radii = this->radii;

        int l, r;
        S::promote(objects, metric, l, r);
        T &parent_l = objects[l], &parent_r = objects[r];
        int parent_distance_l = parent_distances[l], parent_distance_r = parent_distances[r];

        vector<int> objects_l, objects_r;
        S::partition(objects, metric, l, r, objects_l, objects_r);

        // Compute radii of the two new nodes.
        array<int, N> distances_l, distances_r;
        distances_l.fill(0);
        distances_r.fill(0);
        for (int i : objects_l) {
            distances_l[i] = metric->distance(parent_l, objects[i]);
        }
        for (int i : objects_r) {
            distances_r[i] = metric->distance(parent_r, objects[i]);
        }
        int radius_l = INT_MIN;
        for (int d : distances_l) {
            radius_l = max(radius_l, d);
        }
        int radius_r = INT_MIN;
        for (int d : distances_r) {
            radius_r = max(radius_r, d);
        }

        // Fill objects and distances in new nodes.
        reset();
        for (int i : objects_l) {
            add_entry(objects[i], distances_l[i], children[i], radii[i]);
        }
        for (int i : objects_r) {
            new_node->add_entry(objects[i], distances_r[i], children[i], radii[i]);
        }

        if (is_root()) {
            InternalNode<T, S, N> *new_root = new InternalNode<T, S, N>(root, metric);
            new_root->add_entry(parent_l, parent_distance_l, this, radius_l);
            new_root->add_entry(parent_r, parent_distance_r, new_node, radius_r);
            *root = new_root;
        } else {
            parent_node->set_entry(parent_index, parent_l, parent_distance_l, this, radius_l);
            parent_node->add_entry(parent_r, parent_distance_r, new_node, radius_r);
        }
    }

    void set_entry(int entry, T& object, int parent_distance, INode<T, S, N>* child, int radius) {
        valid.set(entry);
        objects[entry] = object;
        parent_distances[entry] = parent_distance;
        children[entry] = child;
        radii[entry] = radius;

        child->set_parent(this, entry);
    }

    void add_entry(T& object, int parent_distance, INode<T, S, N>* child, int radius) {
        if (full()) {
            split();
        }
        for (int i = 0; i < N; i++) {
            if (!valid[i]) {
                set_entry(i, object, parent_distance, child, radius);
            }
            return;
        }
    }

protected:
    void set_parent(INode<T, S, N> *parent, int parent_index) {
        parent_node = parent;
        this->parent_index = parent_index;
    }

private:
    inline bool full() {
        return valid.all();
    }
    
    inline bool is_root() {
        return parent_node == NULL;
    }

    inline void reset() {
        valid.reset();
    }

    INode<T, S, N> *parent_node, **root;
    int parent_index;
    IMetric<T> *metric;

    bitset<N> valid;
    array<T, N> objects;
    array<int, N> parent_distances;
    array<INode<T, S, N>*, N> children;
    array<int, N> radii;
};

template <class T, class S, size_t N> class LeafNode : public INode<T, S, N> {
public:
    LeafNode(INode<T, S, N> **root, IMetric<T> *metric) {
        this->root = root;
        this->metric = metric;
    }

    void range(T &query_object, int query_radius, int parent_query_distance, vector<T> &results) {
        for (int i = 0; i < N; i++) {
            int parent_distance = parent_distances[i];
            T object = objects[i];
            
            if (valid[i] && abs(parent_query_distance - parent_distance) <= query_radius) {
                int query_distance = metric->distance(object, query_object);
                if (query_distance <= query_radius) {
                    results.push_back(object);
                }
            }
        }
    }

    void range(T &query_object, int query_radius, vector<T> &results) {
        range(query_object, query_radius, 0, results);
    }
    
    void insert(T& insert_object, int parent_distance) {
        bool success = false;
        for (int i = 0; i < N; i++) {
            if (!valid[i]) {
                objects[i] = insert_object;
                parent_distances[i] = parent_distance;
                valid.set(i);
                success = true;
                break;
            }
        }
        assert(success && "Insertion failed.");

        if (full()) {
            split();
        }
    }
    
    void split() {
        LeafNode<T, S, N> *new_node = new LeafNode<T, S, N>(root, metric);
        array<T, N> objects = this->objects;
        
        int l, r;
        S::promote(objects, metric, l, r);
        T &parent_l = objects[l], &parent_r = objects[r];
        int parent_distance_l = parent_distances[l], parent_distance_r = parent_distances[r];

        vector<int> objects_l, objects_r;
        S::partition(objects, metric, l, r, objects_l, objects_r);

        // Compute radii of the two new nodes.
        array<int, N> distances_l, distances_r;
        distances_l.fill(0);
        distances_r.fill(0);
        for (int i : objects_l) {
            distances_l[i] = metric->distance(objects[l], objects[i]);
        }
        for (int i : objects_r) {
            distances_r[i] = metric->distance(objects[r], objects[i]);
        }
        int radius_l = INT_MIN;
        for (int d : distances_l) {
            radius_l = max(radius_l, d);
        }
        int radius_r = INT_MIN;
        for (int d : distances_r) {
            radius_r = max(radius_r, d);
        }
        
        // Fill objects and distances in new nodes.
        reset();
        for (int i : objects_l) {
            add_entry(objects[i], distances_l[i], NULL, 0);
        }
        for (int i : objects_r) {
            new_node->add_entry(objects[i], distances_r[i], NULL, 0);
        }

        if (is_root()) {
            InternalNode<T, S, N> *new_root = new InternalNode<T, S, N>(root, metric);
            new_root->add_entry(parent_l, parent_distance_l, this, radius_l);
            new_root->add_entry(parent_r, parent_distance_r, new_node, radius_r);
            *root = new_root;
        } else {
            parent_node->set_entry(parent_index, parent_l, parent_distance_l, this, radius_l);
            parent_node->add_entry(parent_r, parent_distance_r, new_node, radius_r);
        }
    }

protected:
    void set_parent(INode<T, S, N> *parent, int parent_index) {
        parent_node = parent;
        this->parent_index = parent_index;
    }

    void set_entry(int entry, T& object, int parent_distance, INode<T, S, N>* child, int radius) {
        set_entry(entry, object, parent_distance);
    }

    void add_entry(T& object, int parent_distance, INode<T, S, N>* child, int radius) {
        add_entry(object, parent_distance);
    }
    
private:
    inline bool full() {
        return valid.all();
    }

    inline bool is_root() {
        return parent_node == NULL;
    }

    void reset() {
        valid.reset();
    }

    void set_entry(int entry, T& object, int parent_distance) {
        valid.set(entry);
        objects[entry] = object;
        parent_distances[entry] = parent_distance;
    }
        
    void add_entry(T& object, int parent_distance) {
        for (int i = 0; i < N; i++) {
            if (!valid[i]) {
                set_entry(i, object, parent_distance);
            }
            return;
        }
        assert(false && "Adding entry failed.");
    }
    
    INode<T, S, N> *parent_node, **root;
    int parent_index;
    IMetric<T> *metric;
    
    bitset<N> valid;
    array<T, N> objects;
    array<int, N> parent_distances;
};

template <class T, size_t N> class RandomSplit {
public:
    static void promote(array<T, N> objects, IMetric<T> *metric, int &parent_l, int &parent_r) {
        random_device rd;
        parent_l = rd() % N;
        parent_r = rd() % N;
    }

    static void partition(array<T, N> objects,
                          IMetric<T> *metric,
                          int parent_l, int parent_r,
                          vector<int> &objects_l, vector<int> &objects_r) {
        for (int i = 0; i < N; i++) {
            if (objects[i].is_empty() || i == parent_l || i == parent_r) { continue; }

            T &obj = objects[i], &pl = objects[parent_l], &pr = objects[parent_r];
            if (metric->distance(obj, pl) < metric->distance(obj, pr)) {
                objects_l.push_back(i);
            } else {
                objects_r.push_back(i);
            }
        }
    }
};

template <class T, class S, size_t N> class MTree {
public:
    MTree(IMetric<T> *metric) {
        root = new LeafNode<T, S, N>(&root, metric);
    }

    void insert(T& insert_object) {
        root->insert(insert_object);
    }

    void range(T &query_object, int query_radius, vector<T> &results) {
        root->range(query_object, query_radius, results);
    }
    
private:
    INode<T, S, N> *root;
};


