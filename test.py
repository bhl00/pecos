from pecos.xmc import PostProcessor, Indexer, LabelEmbeddingFactory
from pecos.xmc.xlinear import XLinearModel

tmpdir = "../../datasets/saves/tests/"

train_X_file = "../../datasets/eurlex-4k/X.trn.npz"
train_Y_file = "../../datasets/eurlex-4k/Y.trn.npz"
test_X_file = "../../datasets/eurlex-4k/X.tst.npz"
Xt = XLinearModel.load_label_matrix(train_X_file)
Yt = XLinearModel.load_label_matrix(train_Y_file)
model_folder = "../../datasets/saves/tests/"
label_feat = LabelEmbeddingFactory.create(Yt, Xt, method="pifa")

model_folder_list = []
# Obtain xlinear models with vairous number of splits
for splits in [None]:
    model_folder_local = f"{model_folder}-{splits}"
    cluster_chain = Indexer.gen(label_feat, nr_splits=splits)
    py_model = XLinearModel.train(Xt, Yt, C=cluster_chain)
    py_model.save(model_folder_local)
    model_folder_list.append(model_folder_local)

X = XLinearModel.load_label_matrix(test_X_file)
label_size = Yt.shape[1]

select_outputs_list = []
select_outputs_list.append(
    set([0, label_size // 3, label_size // 2, label_size * 2 // 3, label_size - 1])
)
select_outputs_list.append(set([n for n in range(label_size // 10)]))
select_outputs_list.append(set([n for n in range(label_size)]))

def test_on_model(model, X, select_outputs_list):
    for pp in PostProcessor.valid_list():
        # Batch mode topk
        py_sparse_topk_pred = model.predict(X, post_processor=pp)
        py_dense_topk_pred = model.predict(X.todense(), post_processor=pp)

        # Sparse Input
        py_select_sparse_topk_pred = model.predict(
            X, select_outputs_csr=py_sparse_topk_pred, post_processor=pp
        )
        # Dense Input
        py_select_dense_topk_pred = model.predict(
            X.todense(), select_outputs_csr=py_dense_topk_pred, post_processor=pp
        )

        error = py_sparse_topk_pred - py_select_sparse_topk_pred
        assert len(error.data) == 0
        print(f"Passed model:{model_folder_local} (batch, sparse, topk) post_processor:{pp})")
        error = py_dense_topk_pred - py_select_dense_topk_pred
        assert len(error.data) == 0
        print(f"Passed model:{model_folder_local} (batch, dense, topk) post_processor:{pp})")

        # Batch mode select output(symmetric, assymetric, all)
        py_sparse_pred = model.predict(X, only_topk=label_size, post_processor=pp)
        py_dense_pred = model.predict(X.todense(), only_topk=label_size, post_processor=pp)

        for select_output in select_outputs_list:
            true_sparse_pred = py_sparse_pred.copy()
            for i_nnz in range(true_sparse_pred.nnz):
                if true_sparse_pred.indices[i_nnz] not in select_output:
                    true_sparse_pred.data[i_nnz] = 0
            true_sparse_pred.eliminate_zeros()

            true_dense_pred = py_dense_pred.copy()
            for i_nnz in range(true_dense_pred.nnz):
                if true_dense_pred.indices[i_nnz] not in select_output:
                    true_dense_pred.data[i_nnz] = 0
            true_dense_pred.eliminate_zeros()

            # Sparse Input
            py_select_sparse_pred = model.predict(
                X, select_outputs_csr=true_sparse_pred, post_processor=pp
            )
            # Dense Input
            py_select_dense_pred = model.predict(
                X.todense(), select_outputs_csr=true_dense_pred, post_processor=pp
            )

            error = true_sparse_pred - py_select_sparse_pred
            assert len(error.data) == 0
            print(f"Passed model:{model_folder_local} (batch, sparse, select) post_processor:{pp}")
            error = true_dense_pred - py_select_dense_pred
            assert len(error.data) == 0
            print(f"Passed model:{model_folder_local} (batch, dense, select) post_processor:{pp}")

        # Realtime mode topk
        for i in range(X.shape[0]):
            query_slice = X[[i], :]
            query_slice.sort_indices()

            py_sparse_realtime_pred = model.predict(query_slice, post_processor=pp)
            py_dense_realtime_pred = model.predict(query_slice.todense(), post_processor=pp)

            # Sparse Input
            py_select_sparse_realtime_pred = model.predict(
                query_slice, select_outputs_csr=py_sparse_realtime_pred, post_processor=pp
            )
            # Dense input
            py_select_dense_realtime_pred = model.predict(
                query_slice.todense(), select_outputs_csr=py_dense_realtime_pred, post_processor=pp
            )

            error = py_sparse_realtime_pred - py_select_sparse_realtime_pred
            assert len(error.data) == 0
            print(f"Passed model:{model_folder_local} (realtime, sparse, topk) post_processor:{pp}")
            error = py_dense_realtime_pred - py_select_dense_realtime_pred
            assert len(error.data) == 0
            print(f"Passed model:{model_folder_local} (realtime, dense, topk) post_processor:{pp}")

        # Realtime mode select output(symmetric, assymetric, all)
        for i in range(X.shape[0]):
            query_slice = X[[i], :]
            query_slice.sort_indices()

            py_sparse_pred = model.predict(query_slice, only_topk=label_size, post_processor=pp)
            py_dense_pred = model.predict(
                query_slice.todense(), only_topk=label_size, post_processor=pp
            )

            for select_output in select_outputs_list:
                true_sparse_pred = py_sparse_pred.copy()
                for i_nnz in range(true_sparse_pred.nnz):
                    if true_sparse_pred.indices[i_nnz] not in select_output:
                        true_sparse_pred.data[i_nnz] = 0
                true_sparse_pred.eliminate_zeros()

                true_dense_pred = py_dense_pred.copy()
                for i_nnz in range(true_dense_pred.nnz):
                    if true_dense_pred.indices[i_nnz] not in select_output:
                        true_dense_pred.data[i_nnz] = 0
                true_dense_pred.eliminate_zeros()

                # Sparse Input
                py_select_sparse_pred = model.predict(
                    query_slice, select_outputs_csr=true_sparse_pred, post_processor=pp
                )
                # Dense Input
                py_select_dense_pred = model.predict(
                    query_slice.todense(), select_outputs_csr=true_dense_pred, post_processor=pp
                )

                error = true_sparse_pred - py_select_sparse_pred
                assert len(error.data) == 0
                print(f"Passed model:{model_folder_local} (realtime, sparse, select) post_processor:{pp}")
                error = true_dense_pred - py_select_dense_pred
                assert len(error.data) == 0
                print(f"Passed model:{model_folder_local} (realtime, dense, select) post_processor:{pp}")

for model_folder_local in model_folder_list:
    model_f = XLinearModel.load(model_folder_local, is_predict_only=False)
    model_t = XLinearModel.load(
        model_folder_local, is_predict_only=True, weight_matrix_type="CSC"
    )

    test_on_model(model_f, X, select_outputs_list)
    test_on_model(model_t, X, select_outputs_list)
