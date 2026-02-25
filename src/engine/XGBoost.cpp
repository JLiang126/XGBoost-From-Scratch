#include "XGBoost-From-Scratch/include/engine/XGBoost.hpp"

#include <iostream>

XGBoost::XGBoost(int num_trees, float learning_rate, int max_depth, float lambda, float gamma, float min_cover, Loss *objective) : num_trees(num_trees), learning_rate(learning_rate), max_depth(max_depth), lambda(lambda), gamma(gamma), min_cover(min_cover), objective(objective) {}

void XGBoost::train(const DataMatrix &data, const vector<float> &labels)
{
    int num_rows = data.get_num_rows();
    vector<float> curr_preds(num_rows, 0.0f);
    vector<derivative> derivatives(num_rows);

    for (size_t i{0}; i < num_trees; i++)
    {
        for (int j = 0; j < num_rows; j++)
            objective->compute_gradients(curr_preds, labels, derivatives);

        Tree XGTree(max_depth, lambda, gamma, min_cover);
        XGTree.build(data, derivatives);

        for (size_t j{0}; j < num_rows; j++)
            curr_preds[j] += learning_rate * XGTree.predict(data.get_row(j));

        forest.push_back(move(XGTree));

        cout << "Tree" << i << "/" << num_trees << "built" << endl;
    }
    return;
}

float XGBoost::predict(const vector<float> &sample) const
{
    float prediction = 0.0f;
    for (const auto &tree : forest)
        prediction += learning_rate * tree.predict(sample);
    return prediction;
}