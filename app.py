import streamlit as st
import graphviz
import math

# Helper to format distances for display
def format_dist(d):
    if d == float('inf'):
        return "∞"
    if d == -float('inf'):
        return "-∞"
    if isinstance(d, (int, float)) and not math.isinf(d) and not math.isnan(d):
        if abs(d - round(d)) < 1e-9: # Tolerance for float to int conversion
            return str(int(round(d)))
        return f"{d:.2f}" # Format floats to 2 decimal places
    return str(d)

# Bellman-Ford Algorithm
def bellman_ford_algorithm(graph_matrix, num_vertices, start_node_idx):
    distances = [float('inf')] * num_vertices
    if num_vertices > 0 and 0 <= start_node_idx < num_vertices:
        distances[start_node_idx] = 0.0

    current_distances = list(distances)
    for _ in range(num_vertices - 1): 
        previous_distances_this_iteration = list(current_distances)
        for u in range(num_vertices):
            for v in range(num_vertices):
                weight = graph_matrix[u][v]
                if weight != float('inf'): 
                    if previous_distances_this_iteration[u] != float('inf') and \
                       previous_distances_this_iteration[u] + weight < current_distances[v]:
                        current_distances[v] = previous_distances_this_iteration[u] + weight
    
    final_distances = list(current_distances)
    negative_cycle_detected = False
    for _ in range(num_vertices): 
        changed_in_neg_cycle_pass = False
        for u in range(num_vertices):
            for v in range(num_vertices):
                weight = graph_matrix[u][v]
                if weight != float('inf'):
                    if final_distances[u] == -float('inf'):
                        if final_distances[v] != -float('inf'):
                            final_distances[v] = -float('inf')
                            negative_cycle_detected = True
                            changed_in_neg_cycle_pass = True
                    elif final_distances[u] != float('inf') and \
                         final_distances[u] + weight < final_distances[v]:
                        final_distances[v] = -float('inf') 
                        negative_cycle_detected = True
                        changed_in_neg_cycle_pass = True
        if not changed_in_neg_cycle_pass:
            break
    return final_distances, negative_cycle_detected

# Display Augmented Matrix Step (Now only for final state)
def display_final_matrix_state(st_container, label, current_distances, graph_matrix, vertex_labels):
    num_vertices = len(graph_matrix)
    html_table = "<table><thead><tr><th></th>" 
    for v_label in vertex_labels:
        html_table += f"<th><b>{v_label}</b></th>"
    html_table += "</tr></thead><tbody>"
    html_table += "<tr><td><b>Dist:</b></td>" 
    for d_val in current_distances:
        html_table += f'<td style="background-color: black; color: white; text-align: center; padding: 5px;"><b>{format_dist(d_val)}</b></td>'
    html_table += "</tr>"
    for i in range(num_vertices):
        html_table += f'<tr><td style="background-color: black; color: white; text-align: center; padding: 5px;"><b>{vertex_labels[i]}</b><br>{format_dist(current_distances[i])}</td>'
        for j in range(num_vertices):
            html_table += f'<td style="text-align: center; padding: 5px;">{format_dist(graph_matrix[i][j])}</td>'
        html_table += "</tr>"
    html_table += "</tbody></table>"
    st_container.markdown(f"#### {label}:")
    st_container.markdown(html_table, unsafe_allow_html=True)
    st_container.markdown("---")

# Reconstruct All Shortest Paths
def reconstruct_shortest_paths(num_vertices, start_node_idx, end_node_idx, final_distances, graph_matrix):
    if num_vertices == 0: return [], "Граф пуст."
    if final_distances[end_node_idx] == float('inf'): return [], "До конечной вершины нет пути."
    if final_distances[end_node_idx] == -float('inf'): return [], "Конечная вершина достижима через отрицательный цикл (кратчайший путь -∞)."
    all_paths = []
    def find_paths_recursive(current_vertex_idx, path_so_far_reversed):
        current_path_from_start = [current_vertex_idx] + path_so_far_reversed 
        if current_vertex_idx == start_node_idx:
            all_paths.append(list(current_path_from_start)); return
        for u_idx in range(num_vertices): 
            edge_weight = graph_matrix[u_idx][current_vertex_idx] 
            if edge_weight != float('inf') and final_distances[u_idx] != float('inf') and final_distances[u_idx] != -float('inf'): 
                if abs(final_distances[u_idx] + edge_weight - final_distances[current_vertex_idx]) < 1e-9:
                    if u_idx not in current_path_from_start: 
                        find_paths_recursive(u_idx, list(current_path_from_start)) 
    find_paths_recursive(end_node_idx, []) 
    if not all_paths:
        if start_node_idx == end_node_idx and abs(final_distances[start_node_idx]) < 1e-9 : 
             return [[start_node_idx]], f"Стоимость пути: {format_dist(final_distances[end_node_idx])}"
        return [], "Не удалось восстановить путь."
    unique_paths_tuples = set(tuple(p) for p in all_paths)
    deduplicated_paths = [list(p_tuple) for p_tuple in unique_paths_tuples]
    return deduplicated_paths, f"Стоимость пути: {format_dist(final_distances[end_node_idx])}"

# Create Graphviz DOT object
def create_graphviz_dot(paths, vertex_labels, start_node_idx, end_node_idx, graph_matrix, final_distances):
    dot = graphviz.Digraph(comment='Shortest Paths Graph'); dot.attr(rankdir='LR') 
    num_vertices = len(vertex_labels); path_edges = set()
    if paths:
        for path in paths: 
            for i in range(len(path) - 1): path_edges.add((path[i], path[i+1]))
    for i in range(num_vertices):
        label = f"{vertex_labels[i]}\n({format_dist(final_distances[i])})"
        color = "lightgrey"
        if i == start_node_idx: color = "lightblue"
        elif i == end_node_idx: color = "lightgreen"
        if final_distances[i] == -float('inf'): color = "pink"
        dot.node(str(i), label, style="filled", fillcolor=color)
    for r_idx in range(num_vertices):
        for c_idx in range(num_vertices):
            weight = graph_matrix[r_idx][c_idx]
            if weight != float('inf'):
                is_path_edge = (r_idx, c_idx) in path_edges
                edge_color = "blue" if is_path_edge else "gray"; penwidth = "2.0" if is_path_edge else "1.0"
                fontcolor = "blue" if is_path_edge else "black"
                dot.edge(str(r_idx), str(c_idx), label=format_dist(weight), color=edge_color, penwidth=penwidth, fontcolor=fontcolor)
    return dot

def symmetrize_matrix_values(num_vertices, v_labels):
    error_messages = []
    symmetrized_at_least_one_upper_value_present = False
    modification_occurred = False

    for r in range(num_vertices):
        for c in range(r + 1, num_vertices): # Iterate upper triangle (r < c)
            upper_key = f"{r}_{c}"
            lower_key = f"{c}_{r}"

            upper_val_str = st.session_state.get(upper_key, "").strip()
            lower_val_str_current = st.session_state.get(lower_key, "").strip()

            if upper_val_str: 
                symmetrized_at_least_one_upper_value_present = True
                if lower_val_str_current and lower_val_str_current != upper_val_str:
                    error_messages.append(
                        f"Конфликт: ячейка ({v_labels[r]},{v_labels[c]}) = '{upper_val_str}', "
                        f"а ({v_labels[c]},{v_labels[r]}) = '{lower_val_str_current}'. "
                        "Оставьте одну пустой или сделайте их одинаковыми."
                    )
                else: 
                    if lower_val_str_current != upper_val_str: 
                        st.session_state[lower_key] = upper_val_str
                        modification_occurred = True
    
    if error_messages:
        for msg in error_messages:
            st.error(msg)
    elif modification_occurred:
        st.success("Матрица успешно отражена относительно главной диагонали.")
    elif symmetrized_at_least_one_upper_value_present and not modification_occurred :
        st.info("Матрица уже симметрична или нижний треугольник соответствует верхнему.")
    elif not symmetrized_at_least_one_upper_value_present:
        st.info("Нет значений в верхнем треугольнике для отражения.")


# Main Streamlit App
def main():
    st.set_page_config(layout="wide", page_title="Алгоритм Беллмана-Форда")
    st.title("Алгоритм Беллмана-Форда")

    st.header("Параметры графа")
    
    n_default = 3
    if 'n_vertices_val' not in st.session_state:
        st.session_state.n_vertices_val = n_default
    if 'current_n_for_matrix_input' not in st.session_state: 
        st.session_state.current_n_for_matrix_input = -1 
    if 'symmetrize_requested' not in st.session_state:
        st.session_state.symmetrize_requested = False

    def on_n_change_callback():
        new_n = st.session_state.n_vertices_input_key 
        st.session_state.n_vertices_val = new_n
        st.session_state.matrix_input_needs_reset = True

    n_val_widget = st.number_input(
        "Количество вершин (n):", 
        min_value=1, 
        value=st.session_state.n_vertices_val, 
        step=1, 
        key="n_vertices_input_key", 
        on_change=on_n_change_callback
    )
    n = st.session_state.n_vertices_val
    vertex_labels = [f"x{i+1}" for i in range(n)]

    if st.session_state.symmetrize_requested:
        symmetrize_matrix_values(n, vertex_labels)
        st.session_state.symmetrize_requested = False 

    col1, col2 = st.columns(2)
    with col1:
        start_node_idx_val = st.session_state.get('start_node_idx_val', 0)
        current_start_idx = min(start_node_idx_val, n-1) if n > 0 else 0
        start_node_label = st.selectbox("Начальная вершина:", vertex_labels, index=current_start_idx, key="start_node_sb")
    
    with col2:
        end_node_idx_val = st.session_state.get('end_node_idx_val', max(0,n-1))
        current_end_idx = min(end_node_idx_val, n-1) if n > 0 else 0
        end_node_label = st.selectbox("Конечная вершина:", vertex_labels, index=current_end_idx, key="end_node_sb")
    
    st.session_state.start_node_idx_val = vertex_labels.index(start_node_label) if start_node_label in vertex_labels and n > 0 else 0
    st.session_state.end_node_idx_val = vertex_labels.index(end_node_label) if end_node_label in vertex_labels and n > 0 else (max(0, n -1) if n > 0 else 0)

    st.subheader("Матрица весов")
    st.markdown(f"Размер {n}x{n}. Оставьте ячейку пустой или введите 'inf'/'∞' для отсутствия ребра.")

    if st.session_state.get('matrix_input_needs_reset', False) or st.session_state.get('current_n_for_matrix_input', -1) != n:
        old_n_for_cleanup = st.session_state.get('current_n_for_matrix_input', 0)
        if old_n_for_cleanup != n and old_n_for_cleanup > 0 :
                for r_old in range(old_n_for_cleanup):
                    for c_old in range(old_n_for_cleanup):
                        key_to_delete = f"{r_old}_{c_old}"
                        if key_to_delete in st.session_state:
                            del st.session_state[key_to_delete]
        
        for r_idx_init in range(n):
            for c_idx_init in range(n):
                key_to_init = f"{r_idx_init}_{c_idx_init}"
                if key_to_init not in st.session_state: 
                    st.session_state[key_to_init] = ""
        st.session_state.current_n_for_matrix_input = n
        st.session_state.matrix_input_needs_reset = False

    if n > 0:
        header_cols = st.columns(n + 1)
        header_cols[0].markdown("<div style='text-align: center; font-weight: bold; height: 38px; display: flex; align-items: center; justify-content: center;'>&nbsp;</div>", unsafe_allow_html=True)
        for c_idx, label_text in enumerate(vertex_labels):
            header_cols[c_idx + 1].markdown(f"<div style='text-align: center; font-weight: bold; height: 38px; display: flex; align-items: center; justify-content: center;'>{label_text}</div>", unsafe_allow_html=True)

        for r_idx in range(n):
            row_cols = st.columns(n + 1)
            row_cols[0].markdown(f"<div style='text-align: center; font-weight: bold; height: 38px; display: flex; align-items: center; justify-content: center;'>{vertex_labels[r_idx]}</div>", unsafe_allow_html=True)
            for c_idx in range(n):
                row_cols[c_idx + 1].text_input(
                    label=f"{r_idx+1}_{c_idx+1}",  # Hidden label for unique widget
                    key=f"{r_idx}_{c_idx}", 
                    placeholder=""
                )
        
        if st.button("Отразить относительно главной диагонали", key="symmetrize_button"):
            st.session_state.symmetrize_requested = True
            st.rerun()

    else:
        st.write("Матрица не отображается для n=0.")

    adj_matrix = []
    valid_matrix = True
    if n > 0:
        try:
            for r_idx in range(n):
                row_list = []
                for c_idx in range(n):
                    val_str = st.session_state.get(f"{r_idx}_{c_idx}", "").strip()
                    if val_str.lower() in ['inf', 'infinity', '∞', ''] or val_str is None:
                        row_list.append(float('inf'))
                    else: 
                        try:
                            row_list.append(float(val_str))
                        except ValueError:
                            st.error(f"Неверное значение '{val_str}' в матрице ({vertex_labels[r_idx]},{vertex_labels[c_idx]}). Используйте числа, 'inf', или оставьте пустым.")
                            valid_matrix = False; break 
                if not valid_matrix: break
                adj_matrix.append(row_list)
        except KeyError as e: 
            st.error(f"Ошибка доступа к данным матрицы (KeyError): {e}. Попробуйте изменить количество вершин.")
            valid_matrix = False
        except Exception as e:
            st.error(f"Ошибка при обработке матрицы: '{e}'.")
            valid_matrix = False
    elif n == 0: 
        valid_matrix = True

    if not valid_matrix and n > 0:
        st.stop()

    start_node_idx = st.session_state.start_node_idx_val
    end_node_idx = st.session_state.end_node_idx_val
    
    if st.button("Выполнить алгоритм", key="run_button"):
        if n == 0:
            st.warning("Количество вершин равно 0. Алгоритм не может быть выполнен.")
            st.stop()
        if not valid_matrix: 
            st.error("Матрица содержит неверные значения. Пожалуйста, исправьте перед выполнением.")
            st.stop()

        st.header("Результаты алгоритма")
        final_distances, negative_cycle_detected = bellman_ford_algorithm(adj_matrix, n, start_node_idx)

        final_state_display_area = st.container()
        matrix_label = "Конечное состояние кратчайших путей"
        if negative_cycle_detected:
            matrix_label = "Конечное состояние (обнаружен отрицательный цикл, пути могут быть -∞)"
        
        display_final_matrix_state(final_state_display_area, matrix_label, final_distances, adj_matrix, vertex_labels)

        if negative_cycle_detected:
            st.warning("Обнаружен отрицательный цикл в графе!")
        
        paths, cost_message = reconstruct_shortest_paths(n, start_node_idx, end_node_idx, final_distances, adj_matrix)
        
        st.subheader(f"Кратчайшие пути от {vertex_labels[start_node_idx] if n > 0 else 'N/A'} до {vertex_labels[end_node_idx] if n > 0 else 'N/A'}")
        st.write(cost_message)
        if paths:
            for p_idx, path_indices in enumerate(paths):
                path_labels = [vertex_labels[node_i] for node_i in path_indices]
                st.markdown(f"**Путь {p_idx+1}:** {' → '.join(path_labels)}")
        elif n > 0 and final_distances[end_node_idx] != float('inf') and final_distances[end_node_idx] != -float('inf') and not (start_node_idx == end_node_idx and abs(final_distances[start_node_idx]) < 1e-9) :
             st.info("Путь не найден между указанными вершинами с текущей стоимостью.")

        st.subheader("Визуализация графа и кратчайших путей")
        if n > 0 :
            try:
                graph_dot = create_graphviz_dot(paths, vertex_labels, start_node_idx, end_node_idx, adj_matrix, final_distances)
                st.graphviz_chart(graph_dot)
            except Exception as e:
                st.error(f"Ошибка при построении графа: {e}")
    else:
        st.info("Настройте параметры графа и нажмите 'Выполнить алгоритм'.")

if __name__ == "__main__":
    main()
