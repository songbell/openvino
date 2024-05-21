// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/runtime/threading/cpu_message.hpp"
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

namespace ov {
namespace threading {

MessageManage::MessageManage() {}

MessageManage::~MessageManage() {}

void MessageManage::send_message(MessageInfo msg_info) {
    {
        std::lock_guard<std::mutex> lock(_msgMutex);
        _messageQueue.push_back(msg_info);
        // std::cout << "send : " << msg_info.msg_type << ", " << (msg_info.rank.size() > 0 ? msg_info.rank[0] : -1) << "\n";
    }
    _msgCondVar.notify_all();
}

std::vector<MessageInfo> MessageManage::wait_message(int cur_rank, int streams_num) {
    std::vector<MessageInfo> messages_total;
    std::unique_lock<std::mutex> lock(_readMutex);
    _readCondVar.wait(lock, [&] {
        // std::cout << "wait_" << cur_rank << " : " << _readQueue[cur_rank].size() << " / " << streams_num << "\n";
        return _readQueue[cur_rank].size() >= streams_num;
    });
    std::swap(_readQueue[cur_rank], messages_total);
    // std::cout << "wait_" << cur_rank << " " << _readQueue[cur_rank].size() << " end\n";
    return messages_total;
}

void MessageManage::infer_wait() {
    std::unique_lock<std::mutex> lock(_inferMutex);
    _inferCondVar.wait(lock);
}

void MessageManage::reduce_wait(int cur_rank, int streams_num) {
    std::unique_lock<std::mutex> lock(_reduceMutex);
    while (_reduceQueue[cur_rank] < streams_num) {
        // std::cout << "reduce_wait_" << cur_rank << " " << _reduceQueue[cur_rank] << " end\n";
        _reduceCondVar.wait(lock);
    }
    _reduceQueue[cur_rank] = 0;
}

void MessageManage::server_wait(int streams_num) {
    if (!_serverThread.joinable()) {
        _readQueue.assign(streams_num, std::vector<MessageInfo>());
        _reduceQueue.assign(streams_num, 0);
        MsgType msg_type;
        _serverThread = std::thread([&, streams_num]() {
            int count = 0;
            int reduce_count = 0;
            while (!_isServerStopped) {
                std::vector<MessageInfo> msgQueue;
                {
                    // std::cout << "server_wait ........" << _isServerStopped << "\n";
                    std::unique_lock<std::mutex> lock(_msgMutex);
                    while (_messageQueue.empty()) {
                        _msgCondVar.wait(lock);
                    }
                    std::swap(_messageQueue, msgQueue);
                    // std::cout << "server_wait receive: " << msgQueue[0].msg_type << " rank:" << msgQueue[0].rank.size()
                    //           << " / " << msgQueue.size() << "\n";
                }

                for (auto rec_info : msgQueue) {
                    msg_type = rec_info.msg_type;
                    if (msg_type == START_INFER) {
                        Task task = std::move(rec_info.task);
                        task();
                    } else if (msg_type == TP) {
                        // Resend _readQueue that failed last time
                        bool stop = false;
                        while (!stop) {
                            stop = true;
                            for (int i = 0; i < streams_num; i++) {
                                if (_readQueue[i].size() == streams_num) {
                                    stop = false;
                                }
                            }
                            if (!stop) {
                                _readCondVar.notify_all();
                            }
                        }
                        for (int i = 0; i < streams_num; i++) {
                            std::lock_guard<std::mutex> lock(_readMutex);
                            _readQueue[i].push_back(rec_info);
                        }
                        _readCondVar.notify_all();
                    } else if (msg_type == CALL_BACK) {  // CALL_BACK
                        count++;
                        // std::cout << "server_wait CALL_BACK: " << count << "/" << streams_num << "\n";
                        if (count == streams_num) {
                            _inferCondVar.notify_one();
                            count = 0;
                        }
                    } else if (msg_type == REDUCE) {  // REDUCE
                        reduce_count++;
                        // std::cout << "server_wait REDUCE: " << reduce_count << "/" << streams_num << "\n";
                        if (reduce_count == streams_num) {
                            {
                                std::lock_guard<std::mutex> lock(_reduceMutex);
                                _reduceQueue.assign(streams_num, reduce_count);
                            }
                            _reduceCondVar.notify_all();
                            reduce_count = 0;
                        }
                    } else if (msg_type == QUIT) {
                        _isServerStopped = true;
                    }
                }
            }
            // std::cout << "-------- server_wait end ---------\n";
        });
    }
}

void MessageManage::setSubCompileModels(std::vector<std::shared_ptr<ov::ICompiledModel>> models) {
    m_sub_compilemodels = models;
}

std::vector<std::shared_ptr<ov::ICompiledModel>> MessageManage::getSubCompileModels() {
    return m_sub_compilemodels;
}

void MessageManage::setSubInferRequest(std::vector<std::shared_ptr<IAsyncInferRequest>> requests) {
    m_sub_infer_requests = requests;
}

std::vector<std::shared_ptr<ov::IAsyncInferRequest>> MessageManage::getSubInferRequest() {
    return m_sub_infer_requests;
}

void MessageManage::stop_server_thread() {
    MessageInfo msg_info;
    msg_info.msg_type = ov::threading::MsgType::QUIT;
    send_message(msg_info);
    if (_serverThread.joinable()) {
        _serverThread.join();
    }
}

void MessageManage::clear() {
    for (size_t i = 0; i < m_sub_infer_requests.size(); i++) {
        std::cout << "infer count_" << i << " : " << m_sub_infer_requests[i].use_count() << "\n";
    }
    m_sub_infer_requests.clear();
    for (size_t i = 0; i < m_sub_compilemodels.size(); i++) {
        std::cout << "model count_" << i << " : " << m_sub_compilemodels[i].use_count() << "\n";
    }
    m_sub_compilemodels.clear();
}

namespace {

class MessageManageHolder {
    std::mutex _mutex;
    std::weak_ptr<MessageManage> _manager;

public:
    MessageManageHolder(const MessageManageHolder&) = delete;
    MessageManageHolder& operator=(const MessageManageHolder&) = delete;

    MessageManageHolder() = default;

    std::shared_ptr<ov::threading::MessageManage> get() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto manager = _manager.lock();
        if (!manager) {
            _manager = manager = std::make_shared<MessageManage>();
        }
        return manager;
    }
};

}  // namespace

std::shared_ptr<MessageManage> message_manager() {
    static MessageManageHolder message_manage;
    auto a = message_manage.get();
    std::cout << "message: " << message_manage.get() << std::endl;
    return a;
}

}  // namespace threading
}  // namespace ov