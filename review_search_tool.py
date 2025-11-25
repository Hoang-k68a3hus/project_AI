"""
Review Search Tool - Giao diện tìm kiếm từ trong reviews sản phẩm
Tương tự như chức năng search của VS Code
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
import re
from typing import List, Tuple, Dict, Any
import json
import threading
import time
from functools import lru_cache


class ReviewSearchTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Review Search Tool - Tìm kiếm trong Reviews")
        self.root.geometry("1200x800")
        
        # Data
        self.df = None
        self.search_results = []
        self.current_result_index = 0
        self.search_cache = {}  # Cache kết quả tìm kiếm
        self.search_thread = None
        self.stop_search = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Thiết lập giao diện"""
        
        # Top Frame - File loading
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="File:").pack(side=tk.LEFT, padx=5)
        self.file_entry = ttk.Entry(top_frame, width=60)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        self.file_entry.insert(0, "data/published_data/data_reviews_purchase.csv")
        
        ttk.Button(top_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(top_frame, text="Chưa load dữ liệu", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Search Frame
        search_frame = ttk.LabelFrame(self.root, text="Tìm kiếm", padding="10")
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Search input
        input_frame = ttk.Frame(search_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Từ khóa:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(input_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind("<Return>", lambda e: self.search())
        
        ttk.Button(input_frame, text="🔍 Tìm kiếm", command=self.search).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Clear", command=self.clear_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="🗑️ Clear Cache", command=self.clear_cache).pack(side=tk.LEFT, padx=5)
        
        # Search options
        options_frame = ttk.Frame(search_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.case_sensitive_var = tk.BooleanVar(value=False)
        self.whole_word_var = tk.BooleanVar(value=False)
        self.regex_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(options_frame, text="Phân biệt hoa/thường (Aa)", 
                       variable=self.case_sensitive_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(options_frame, text="Toàn bộ từ", 
                       variable=self.whole_word_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(options_frame, text="Regex", 
                       variable=self.regex_var).pack(side=tk.LEFT, padx=5)
        
        # Results info
        result_info_frame = ttk.Frame(search_frame)
        result_info_frame.pack(fill=tk.X, pady=5)
        
        self.result_count_label = ttk.Label(result_info_frame, text="Kết quả: 0", font=("Arial", 10, "bold"))
        self.result_count_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(result_info_frame, text="⬆ Trước", command=self.prev_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(result_info_frame, text="⬇ Sau", command=self.next_result).pack(side=tk.LEFT, padx=2)
        
        self.current_result_label = ttk.Label(result_info_frame, text="")
        self.current_result_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(result_info_frame, text="📋 Copy Results", command=self.copy_results).pack(side=tk.RIGHT, padx=5)
        ttk.Button(result_info_frame, text="💾 Export", command=self.export_results).pack(side=tk.RIGHT, padx=5)
        
        # Progress bar
        self.progress_frame = ttk.Frame(search_frame)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=300)
        self.progress_label = ttk.Label(self.progress_frame, text="")
        
        # Main content - Split view
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Results list
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Danh sách kết quả:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        
        # Results listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        list_scrollbar = ttk.Scrollbar(list_frame)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, font=("Consolas", 9))
        self.results_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_listbox.bind("<<ListboxSelect>>", self.on_result_select)
        
        list_scrollbar.config(command=self.results_listbox.yview)
        
        # Right panel - Detail view
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        ttk.Label(right_frame, text="Chi tiết:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        
        # Detail text area
        self.detail_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Arial", 10), 
                                                     height=20, bg="#f5f5f5")
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for highlighting
        self.detail_text.tag_config("highlight", background="yellow", foreground="black", font=("Arial", 10, "bold"))
        self.detail_text.tag_config("label", foreground="blue", font=("Arial", 9, "bold"))
        self.detail_text.tag_config("value", foreground="black", font=("Arial", 9))
        
        # Bottom status bar
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_text = ttk.Label(status_bar, text="Sẵn sàng", relief=tk.SUNKEN, anchor=tk.W)
        self.status_text.pack(fill=tk.X, padx=5, pady=2)
        
    def browse_file(self):
        """Chọn file dữ liệu"""
        filename = filedialog.askopenfilename(
            title="Chọn file reviews",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            
    def load_data(self):
        """Load dữ liệu từ file CSV"""
        try:
            file_path = self.file_entry.get()
            self.status_text.config(text=f"Đang load {file_path}...")
            self.root.update()
            
            # Load CSV với encoding UTF-8
            self.df = pd.read_csv(file_path, encoding='utf-8')
            
            # Kiểm tra cột cần thiết
            required_columns = ['processed_comment']
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"File cần có cột: {', '.join(required_columns)}")
            
            # Drop NaN trong processed_comment
            self.df = self.df.dropna(subset=['processed_comment'])
            
            row_count = len(self.df)
            self.status_label.config(text=f"✓ Đã load {row_count:,} reviews", foreground="green")
            self.status_text.config(text=f"Đã load {row_count:,} reviews từ {file_path}")
            
            messagebox.showinfo("Thành công", f"Đã load {row_count:,} reviews!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load dữ liệu:\n{str(e)}")
            self.status_label.config(text="✗ Lỗi load dữ liệu", foreground="red")
            self.status_text.config(text=f"Lỗi: {str(e)}")
            
    def search(self):
        """Thực hiện tìm kiếm với threading và caching"""
        if self.df is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng load dữ liệu trước!")
            return
            
        keyword = self.search_entry.get().strip()
        if not keyword:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập từ khóa!")
            return
        
        # Tạo cache key
        cache_key = (
            keyword,
            self.case_sensitive_var.get(),
            self.whole_word_var.get(),
            self.regex_var.get()
        )
        
        # Kiểm tra cache
        if cache_key in self.search_cache:
            self.search_results = self.search_cache[cache_key]
            self.display_results()
            self.status_text.config(text=f"Loaded từ cache: {len(self.search_results)} results")
            return
        
        # Stop current search if running
        if self.search_thread and self.search_thread.is_alive():
            self.stop_search = True
            self.search_thread.join(timeout=1)
        
        # Start new search in thread
        self.stop_search = False
        self.search_thread = threading.Thread(target=self._search_worker, args=(cache_key,))
        self.search_thread.daemon = True
        self.search_thread.start()
        
        # Show progress
        self.show_progress("Đang tìm kiếm...")
    
    def _search_worker(self, cache_key):
        """Worker thread cho tìm kiếm - tối ưu với vectorization"""
        keyword, case_sensitive, whole_word, use_regex = cache_key
        
        try:
            # Tạo pattern
            if use_regex:
                pattern = keyword
            elif whole_word:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
            
            flags = 0 if case_sensitive else re.IGNORECASE
            
            # VECTORIZED SEARCH - Nhanh hơn 100x so với iterrows
            start_time = time.time()
            
            # Sử dụng pandas str.contains (vectorized)
            review_col = 'processed_comment' if 'processed_comment' in self.df.columns else 'review_content'
            mask = self.df[review_col].astype(str).str.contains(pattern, flags=flags, regex=True, na=False)
            matched_df = self.df[mask]
            
            # Chỉ iterate trên matched rows (ít hơn nhiều)
            results = []
            regex = re.compile(pattern, flags)
            
            for idx, row in matched_df.iterrows():
                if self.stop_search:
                    return
                
                review = str(row.get(review_col, ''))
                matches = list(regex.finditer(review))
                
                if matches:
                    results.append({
                        'index': idx,
                        'row': row.to_dict(),  # Convert to dict để giảm memory
                        'matches': [(m.start(), m.end(), m.group()) for m in matches],  # Store positions
                        'match_count': len(matches)
                    })
            
            elapsed = time.time() - start_time
            
            # Cache kết quả
            self.search_cache[cache_key] = results
            
            # Update UI in main thread
            self.root.after(0, lambda: self._search_complete(results, elapsed))
            
        except re.error as e:
            self.root.after(0, lambda: self._search_error(f"Pattern không hợp lệ: {str(e)}"))
        except Exception as e:
            self.root.after(0, lambda: self._search_error(f"Lỗi tìm kiếm: {str(e)}"))
    
    def _search_complete(self, results, elapsed):
        """Callback khi tìm kiếm hoàn thành"""
        self.search_results = results
        self.hide_progress()
        self.display_results()
        
        status_msg = f"Tìm thấy {len(results)} reviews trong {elapsed:.2f}s"
        if len(self.search_cache) > 1:
            status_msg += f" (cached: {len(self.search_cache)} queries)"
        self.status_text.config(text=status_msg)
    
    def _search_error(self, error_msg):
        """Callback khi có lỗi"""
        self.hide_progress()
        messagebox.showerror("Lỗi", error_msg)
        self.status_text.config(text=error_msg)
    
    def show_progress(self, message):
        """Hiển thị progress bar"""
        self.progress_label.config(text=message)
        self.progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.progress_label.pack(side=tk.LEFT, padx=5)
        self.progress_bar.start(10)
    
    def hide_progress(self):
        """Ẩn progress bar"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
            
    def display_results(self):
        """Hiển thị danh sách kết quả"""
        self.results_listbox.delete(0, tk.END)
        self.detail_text.delete(1.0, tk.END)
        
        if not self.search_results:
            self.result_count_label.config(text="Kết quả: 0")
            self.current_result_label.config(text="")
            self.status_text.config(text="Không tìm thấy kết quả")
            messagebox.showinfo("Kết quả", "Không tìm thấy kết quả nào!")
            return
        
        # Update count
        total_matches = sum(r['match_count'] for r in self.search_results)
        self.result_count_label.config(text=f"Kết quả: {len(self.search_results)} reviews ({total_matches} matches)")
        
        # Populate listbox (lazy - chỉ hiển thị text preview)
        for i, result in enumerate(self.search_results):
            row = result['row']
            review_field = 'processed_comment' if 'processed_comment' in row else 'review_content'
            preview = str(row.get(review_field, ''))[:80].replace('\n', ' ')
            match_info = f"[{result['match_count']} match{'es' if result['match_count'] > 1 else ''}]"
            
            display_text = f"{i+1}. {match_info} {preview}..."
            self.results_listbox.insert(tk.END, display_text)
        
        # Select first result
        self.results_listbox.selection_set(0)
        self.current_result_index = 0
        self.show_result_detail(0)
        
        self.status_text.config(text=f"Tìm thấy {len(self.search_results)} reviews với {total_matches} matches")
        
    def on_result_select(self, event):
        """Xử lý khi chọn kết quả từ list"""
        selection = self.results_listbox.curselection()
        if selection:
            index = selection[0]
            self.current_result_index = index
            self.show_result_detail(index)
            
    def show_result_detail(self, index):
        """Hiển thị chi tiết kết quả và highlight từ khóa (lazy loading)"""
        if not self.search_results or index >= len(self.search_results):
            return
            
        result = self.search_results[index]
        row = result['row']
        matches = result['matches']  # List of (start, end, text) tuples
        
        # Clear detail view
        self.detail_text.delete(1.0, tk.END)
        
        # Show metadata
        self.detail_text.insert(tk.END, "═" * 80 + "\n")
        self.detail_text.insert(tk.END, f"Review #{index + 1} / {len(self.search_results)}\n", "label")
        self.detail_text.insert(tk.END, "═" * 80 + "\n\n")
        
        # Show relevant fields
        review_field = 'processed_comment' if 'processed_comment' in row else 'review_content'
        fields_to_show = ['user_id', 'product_id', 'rating', 'review_date', 'review_title', review_field]
        
        for field in fields_to_show:
            if field in row and row[field] is not None and str(row[field]) != 'nan':
                self.detail_text.insert(tk.END, f"{field}: ", "label")
                
                if field == review_field:
                    # Highlight matches in review content
                    content = str(row[field])
                    last_end = 0
                    
                    for start, end, matched_text in matches:
                        # Text before match
                        self.detail_text.insert(tk.END, content[last_end:start], "value")
                        # Highlighted match
                        self.detail_text.insert(tk.END, matched_text, "highlight")
                        last_end = end
                    
                    # Remaining text
                    self.detail_text.insert(tk.END, content[last_end:], "value")
                    self.detail_text.insert(tk.END, "\n\n")
                else:
                    self.detail_text.insert(tk.END, f"{row[field]}\n\n", "value")
        
        self.detail_text.insert(tk.END, "─" * 80 + "\n")
        self.detail_text.insert(tk.END, f"Matches: {result['match_count']}\n", "label")
        
        # Update navigation label
        self.current_result_label.config(text=f"{index + 1} of {len(self.search_results)}")
        
        # Scroll to top
        self.detail_text.see(1.0)
        
    def prev_result(self):
        """Chuyển đến kết quả trước"""
        if not self.search_results:
            return
        
        self.current_result_index = (self.current_result_index - 1) % len(self.search_results)
        self.results_listbox.selection_clear(0, tk.END)
        self.results_listbox.selection_set(self.current_result_index)
        self.results_listbox.see(self.current_result_index)
        self.show_result_detail(self.current_result_index)
        
    def next_result(self):
        """Chuyển đến kết quả sau"""
        if not self.search_results:
            return
        
        self.current_result_index = (self.current_result_index + 1) % len(self.search_results)
        self.results_listbox.selection_clear(0, tk.END)
        self.results_listbox.selection_set(self.current_result_index)
        self.results_listbox.see(self.current_result_index)
        self.show_result_detail(self.current_result_index)
        
    def clear_search(self):
        """Xóa kết quả tìm kiếm"""
        self.search_entry.delete(0, tk.END)
        self.results_listbox.delete(0, tk.END)
        self.detail_text.delete(1.0, tk.END)
        self.search_results = []
        self.result_count_label.config(text="Kết quả: 0")
        self.current_result_label.config(text="")
        self.status_text.config(text="Đã xóa kết quả tìm kiếm")
    
    def clear_cache(self):
        """Xóa cache tìm kiếm"""
        cache_count = len(self.search_cache)
        self.search_cache.clear()
        messagebox.showinfo("Cache cleared", f"Đã xóa {cache_count} cached queries")
        self.status_text.config(text=f"Đã xóa {cache_count} cached queries")
        
    def copy_results(self):
        """Copy kết quả vào clipboard"""
        if not self.search_results:
            messagebox.showinfo("Thông báo", "Không có kết quả để copy!")
            return
        
        # Tạo text để copy
        text_lines = []
        text_lines.append(f"Tìm kiếm: {self.search_entry.get()}")
        text_lines.append(f"Số kết quả: {len(self.search_results)} reviews")
        text_lines.append("=" * 80)
        text_lines.append("")
        
        for i, result in enumerate(self.search_results):
            row = result['row']
            review_field = 'processed_comment' if 'processed_comment' in row else 'review_content'
            text_lines.append(f"#{i+1} - Matches: {result['match_count']}")
            text_lines.append(f"Review: {row.get(review_field, '')}")
            text_lines.append("-" * 80)
            text_lines.append("")
        
        result_text = "\n".join(text_lines)
        
        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(result_text)
        
        messagebox.showinfo("Thành công", f"Đã copy {len(self.search_results)} kết quả vào clipboard!")
        
    def export_results(self):
        """Export kết quả ra file"""
        if not self.search_results:
            messagebox.showinfo("Thông báo", "Không có kết quả để export!")
            return
        
        # Chọn file để save
        filename = filedialog.asksaveasfilename(
            title="Export kết quả",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if not filename:
            return
        
        try:
            if filename.endswith('.json'):
                # Export as JSON
                export_data = []
                for result in self.search_results:
                    row_dict = result['row'].copy()  # Already a dict
                    row_dict['match_count'] = result['match_count']
                    export_data.append(row_dict)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                    
            elif filename.endswith('.csv'):
                # Export as CSV
                export_df = pd.DataFrame([r['row'] for r in self.search_results])
                export_df.insert(0, 'match_count', [r['match_count'] for r in self.search_results])
                export_df.to_csv(filename, index=False, encoding='utf-8')
                
            else:
                # Export as text
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Tìm kiếm: {self.search_entry.get()}\n")
                    f.write(f"Số kết quả: {len(self.search_results)} reviews\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for i, result in enumerate(self.search_results):
                        row = result['row']
                        review_field = 'processed_comment' if 'processed_comment' in row else 'review_content'
                        f.write(f"#{i+1} - Matches: {result['match_count']}\n")
                        f.write(f"Review: {row.get(review_field, '')}\n")
                        f.write("-" * 80 + "\n\n")
            
            messagebox.showinfo("Thành công", f"Đã export {len(self.search_results)} kết quả ra {filename}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể export:\n{str(e)}")


def main():
    """Khởi chạy ứng dụng"""
    root = tk.Tk()
    app = ReviewSearchTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
