/*
 * @Author: superboy
 * @Date: 2024-06-12 16:37:54
 * @LastEditTime: 2024-06-13 21:42:26
 * @LastEditors: superboy
 * @Description: Just for multithread test!
 * @FilePath: /gem5-rvm/tests/cwq/pthread_test/pthread_test.c
 * 
 */

#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
int a = 0;  // 将变量 'a' 移到全局作用域，以便所有线程都能访问

void* worker(void* arg) {
    pthread_mutex_lock(&lock);  // 加锁，确保同一时间只有一个线程修改变量 'a'
    printf("Thread %ld: holding the lock\n", (long)arg);
    a = a + 1;  // 修改全局变量 'a'
    pthread_mutex_unlock(&lock);  // 解锁
    return NULL;
}

int main() {
    pthread_t t1, t2;

    pthread_create(&t1, NULL, worker, (void*)1);
    pthread_create(&t2, NULL, worker, (void*)2);

    pthread_join(t1, NULL);  // 等待线程 t1 完成
    pthread_join(t2, NULL);  // 等待线程 t2 完成

    printf("Final value of 'a': %d\n", a);  // 打印最终的 'a' 值，预期为 2

    pthread_mutex_destroy(&lock);  // 销毁互斥锁
    return 0;
}

